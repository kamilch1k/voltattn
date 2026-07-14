# Quantizing your KV cache can make llama.cpp *slower* — measured on V100 and RTX 4080

People quantize the KV cache to fit longer contexts on cheap GPUs. The
assumption is that fewer bytes also means faster decode — attention at
batch 1 is bandwidth-bound, so half the bytes should be up to twice the
speed. Measured end to end, on current llama.cpp, the opposite happens:
**a quantized KV cache decodes tokens *slower* than fp16**, mildly on Ada,
severely on Volta. This note shows the measurements, the kernel-level
reason, and — via this repo — how much speed is actually available.

Everything here is reproducible: exact commit, hardware, and commands are
at the bottom. Raw logs are in [`results/`](../results/).

## End to end: tokens/s, real model

`llama-bench`, Qwen2.5-1.5B-Instruct Q4_0 (GQA 6, head size 128), FA on,
tg32 at three context depths. llama.cpp build `14d3ba4` (2026-07-14).

**Tesla V100-FHHL-16GB (sm_70):**

| KV cache | d=0 | d=4096 | d=16384 |
|----------|------|--------|---------|
| f16      | 256.4 | 248.2 | **203.5** |
| q8_0     | 245.4 | 221.5 | 169.9 (−17%) |
| q4_0     | 244.5 | 213.3 | **159.7 (−22%)** |

**RTX 4080 Laptop (sm_89):**

| KV cache | d=0 | d=4096 | d=16384 |
|----------|------|--------|---------|
| f16      | 269.5 | 209.5* | **191.4** |
| q8_0     | 252.3 | 237.2 | 189.3 (−1%) |
| q4_0     | 263.8 | 237.1 | **177.9 (−7%)** |

*\*noisy (±41, laptop DVFS); other rows ±1–13.*

On the V100, quantizing the cache costs speed at **every** depth, and the
more you compress, the more you lose: q4_0 gives up 22% of decode speed at
16K context while using 3.6× less VRAM. On Ada, q8_0 is a wash and q4_0 is
a mild net loss at long context. In no measured configuration did
quantization make decode *faster* — the thing the byte math promises.

## Op level: where the time goes

`test-backend-ops perf -o FLASH_ATTN_EXT`, batch-1 decode case
(kv=7680, head size 64, GQA 8). "Effective GB/s" counts the *compressed*
K+V payload divided by time — the useful bandwidth the kernel achieves.

| GPU | K/V type | µs/run | effective GB/s | % of measured ceiling |
|-----|----------|--------|----------------|------------------------|
| V100 | f16  | 36.4  | 432.3 | 53% |
| V100 | q8_0 | 250.1 | 33.4  | 4.1% |
| V100 | q4_0 | 249.8 | 17.7  | **2.2%** |
| 4080L | f16  | 69.4 | 226.6 | 54% |
| 4080L | q8_0 | 81.2 | 102.9 | 24% |
| 4080L | q4_0 | 81.7 | 54.2  | 13% |

(Ceilings measured with this repo's `--probe`: 809 GB/s V100,
421.5 GB/s 4080 Laptop. Ada f16 rows partially L2-served; V100's 6 MB L2
cannot cache these working sets, so its numbers are honest DRAM.)

Two different failure modes:

**On Volta the quantized op is 6.9× slower than f16** — and q4_0 and q8_0
cost the *same* 250 µs. That flat cost is the signature of a fixed extra
pass: llama.cpp's kernel dispatch on Volta routes GQA-model decode to the
tile kernel (`fattn.cu`: the vec kernel is only chosen when
`Q->ne[1] * gqa_ratio_eff <= 2`), and the tile kernel requires f16 K/V
(`need_f16_K/V` in `ggml_cuda_flash_attn_ext_get_alloc_size`). So the
engine **re-dequantizes the entire quantized KV cache into an f16 scratch
buffer for every generated token**. The quantized bytes are read, expanded
to f16 (written, then read back) — more traffic than just storing f16,
plus the conversion kernels themselves.

**On Ada the vec kernel does read quantized data directly** — no
conversion pass — but it still reaches only 13–24% of bandwidth, and q4_0
is slower than f16 in absolute time (81.7 µs vs 69.4 µs) despite reading
4.4× fewer bytes. Fewer bytes per position with a fixed per-position cost
means the memory system starves — the same effect this repo measures and
fixes in its own kernels.

## What's actually available

This repo's restructured kernels, on the **same rented V100**, same
methodology (`--bench`, H=32, D=128, L=16384): q8+ and q4+ run at
**2.35× the fp16 baseline** (67% / 35% MBU), and the un-quantized control
f16+ holds 97–99% MBU. Against llama.cpp's current 0.78× (q4_0, 16K, e2e)
the attention path has roughly **3× on the table** on exactly the hardware
whose users most need quantized caches — the $0.10/hr V100s and the P40s
that r/LocalLLaMA runs on.

None of this is a criticism of llama.cpp — Volta is a fallback path in a
codebase that optimizes primarily for current hardware, and the vec kernel
family is excellent where it's tuned. It's a measurement of an
opportunity: decode-time attention with quantized KV on pre-Turing (and
even Ada) hardware is latency-structured, not byte-structured, and
restructuring it pays byte-ratio speedups instead of byte-ratio slowdowns.

## Reproduce

```bash
# llama.cpp @ 14d3ba4, CUDA 12.4, V100 (sm_70)
cmake -B build -S . -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-bench test-backend-ops

./build/bin/test-backend-ops perf -o FLASH_ATTN_EXT -b CUDA0
./build/bin/llama-bench -m qwen2.5-1.5b-instruct-q4_0.gguf -fa on -p 0 -n 32 \
    -d 0,4096,16384 -ctk f16  -ctv f16  -r 3 -o md
# repeat with -ctk q8_0 -ctv q8_0 and -ctk q4_0 -ctv q4_0

# this repo's kernels + ceiling probe on the same box:
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=70 && cmake --build build
./build/voltattn --probe && ./build/voltattn --bench
```

Caveats, stated honestly: the model is small (1.5B), so attention is only
part of each token — the op-level gap (6.9×) is much larger than the
end-to-end gap (−22%); with bigger models or longer contexts the
end-to-end cost grows. The op-level quantized comparison uses the one
quantized shape in llama.cpp's default perf set (kv=7680, hs=64, gqa=8).
Effective-bandwidth numbers count compressed payload only; the Volta path
physically moves ~3× more than that. V100 card is the 150 W FHHL variant.
