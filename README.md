# voltattn

[![CI](https://github.com/kamilch1k/voltattn/actions/workflows/ci.yml/badge.svg)](https://github.com/kamilch1k/voltattn/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**A fused dequant + attention decode kernel, written for — and measured on — sm_70 (Volta / V100).**

A quantized KV cache is a *memory* win. Batch-1 decode attention is
*bandwidth*-bound. The only way the first becomes a *speed* win is a kernel
that streams the compressed bytes and dequantizes in-register — never
materializing the fp16 cache in global memory:

```
score_j = q · dequant(K_j) / sqrt(D)      (online softmax, one pass)
out     = Σ_j softmax_j · dequant(V_j)
```

INT8 halves the KV bytes of fp16; INT4 quarters them. **If the fused kernel
holds the fp16 baseline's MBU, tokens/s scales with the compression ratio** —
that is the claim this repo tests, with the fp16 baseline in the same binary
and every number measured against an empirically probed bandwidth ceiling.
Same methodology as [membound](https://github.com/kamilch1k/membound), applied
to the attention path.

## Why sm_70, specifically

The target device is V100 — no FP8, no bf16, no `cp.async`, no TMA, first-gen
tensor cores (unused here; this kernel is load-dominated). Everything in the
kernel is Volta-native by construction: plain coalesced per-lane loads,
`__shfl_xor_sync` reductions, fp32 accumulate, `__expf`.

Two Volta-specific design pressures actually shaped the code:

1. **The compute-to-bandwidth ratio is ~4× tighter than Ada.** An RTX 4080
   gives ~70 FLOPs per byte of DRAM bandwidth; a V100 gives ~18. Dequant
   arithmetic that is effectively free on Ada can throttle a V100 kernel — so
   the unpack path is bit-ops + one convert + one FMA per element, with the
   group scale staged once per lane, not recomputed.
2. **CUDA 13 removed sm_70 offline compilation.** Volta builds need a CUDA
   12.x toolchain — which is what V100 cloud boxes ship. CI compiles this repo
   with `-DCMAKE_CUDA_ARCHITECTURES=70` under CUDA 12.4 on every push, so
   Volta-buildability is continuously proven even though development happens
   on an Ada laptop.

## The finding: compression alone didn't pay — memory-level parallelism did

First-cut kernels (one warp per position, 32 lanes across D) hold ~95% MBU at
fp16 — but collapse to 25–60% MBU when quantized. Fewer bytes per position
means the fixed per-position cost (a 5-step warp reduce + the dependent
online-softmax update) dominates, and the memory system starves. At L=64K,
INT4 was *slower* than fp16 — compression fully wasted.

The `q8+`/`q4+` kernels restructure to 8 lanes per position with 4 positions
in flight per warp: 4× the loads issued per iteration (128-bit loads for
INT8), a 3-step reduce, and four independent softmax chains for the scheduler
to interleave. Measured on the development GPU (RTX 4080 Laptop, sm_89,
421.5 GB/s measured read ceiling), H=32 heads, D=128:

| fmt | L | best ms | GB/s | MBU | tok/s | vs f16 |
|-----|-------|---------|-------|-------|-------|--------|
| f16 | 16384 | 0.675 | 397.8 | 94.4% | 1482 | 1.00× |
| q8 (naive) | 16384 | 0.556 | 248.9 | 59.1% | 1798 | 1.21× |
| **q8+** | 16384 | **0.339** | **408.4** | **96.9%** | **2950** | **1.99×** |
| q4 (naive) | 16384 | 0.599 | 119.0 | 28.2% | 1670 | 1.13× |
| **q4+** | 16384 | **0.278** | **256.3** | **60.8%** | **3595** | **2.43×** |

At L=4096: f16 5711 tok/s → q8+ 11097 (1.94×) → q4+ 13754 (2.41×).
At L=65536: q8+ holds **98.7% MBU (2.02×)** — INT8's byte ratio is 1.94×, so
INT8 compression converts to speed at effectively full efficiency.

Both kernel generations stay in the binary — the delta *is* the point.

**Remaining headroom, stated honestly:** q4+ sits at ~58–61% MBU (2.4× of a
theoretical 3.76×) — the residual bottleneck is the nibble-unpack ALU chain
and per-position scale loads; the next levers are wider per-lane tiles and
staging scales for a tile of positions. At L=64K q4+ dips to ~49% MBU —
split-count tuning territory. The harness re-probes the ceiling per device;
the V100 table below came from the same `--bench` binary.

## Measured on the target: V100

Rented V100 box (Tesla V100-FHHL-16GB, sm_70, 80 SMs, CUDA 12.4 toolchain),
measured read ceiling **809 GB/s**. All 30 selftest cases pass on real Volta.
Same harness, H=32, D=128, L=16384:

| fmt | L | best ms | GB/s | MBU | tok/s | vs f16 |
|-----|-------|---------|-------|-------|-------|--------|
| f16 | 16384 | 0.597 | 449.6 | 55.6% | 1675 | 1.00× |
| q8 (naive) | 16384 | 0.606 | 228.3 | 28.2% | 1650 | 0.98× |
| **q8+** | 16384 | **0.256** | **540.7** | **66.8%** | **3906** | **2.33×** |
| q4 (naive) | 16384 | 0.750 | 95.1 | 11.8% | 1334 | 0.80× |
| **q4+** | 16384 | **0.253** | **281.9** | **34.8%** | **3954** | **2.36×** |

At L=4096: f16 6341 tok/s → q8+ 13378 (2.11×) → q4+ 13563 (2.14×).
At L=65536: q8+ holds 571.5 GB/s (**2.41×**); q4+ reaches **2.75×**.

Volta makes the repo's point more sharply than Ada does:

- **Naive compression loses outright on the target device.** q8 is 0.98× — a
  wash; q4 is 0.80× — a 20% *slowdown* from quartering the bytes. Quantizing
  the KV cache without restructuring the kernel is worse than doing nothing.
- **The fp16 baseline itself is latency-limited on Volta**: ~53–57% MBU
  against the probed ceiling, where the identical kernel holds 94% on Ada.
  One position per warp doesn't put enough loads in flight for V100's
  latency. So the restructure pays twice — q8+ at L=64K runs 2.41× against a
  1.94× byte ratio, out-running compression because 4-positions-in-flight
  also supplies the latency hiding the baseline lacks.
- The obvious next kernel is **f16+** — the same 4-position restructure on
  the uncompressed cache — to split the compression win from the
  memory-level-parallelism win cleanly.

(Card caveat: FHHL is the 150 W single-slot V100; SXM2 modules run 250–300 W
with higher clocks, which should lift the latency-bound numbers.)

## Correctness (gated, not asserted)

`voltattn` (no arguments) runs 30 cases — 5 formats × 6 shapes, including
odd lengths (129/257/513/1000), both head dims (64/128), and multi-split
paths — each checked against a **double-precision CPU reference computed on
the exactly-dequantized values** (`q·s` at fp16-scale precision). The gate
verifies kernel arithmetic and indexing; quantization error itself is a
modeling choice, reported by the quantizer, not smuggled into tolerance.

The split-merge path (grid split over KV length + log-sum-exp combine) is the
numerically delicate part; it is exercised by every `S>1` case.

## Quantization formats

Deliberately standard-shaped, clean-room:

- **q8** — symmetric per-group scale (group = 64 dims along D), 1 B/elem +
  fp16 scale per group.
- **q4** — symmetric per-group scale, q ∈ [−7,7] stored as `(q+8)` nibbles,
  2 elems/byte.

Per-token or per-channel variants are drop-in swaps of the `quantize()` /
scale-indexing pair; the kernel structure doesn't change.

## Run it

```bash
# any CUDA GPU (dev): correctness gate
cmake -B build -S . && cmake --build build
./build/voltattn            # 30-case selftest vs double reference
./build/voltattn --bench    # ceiling probe + full benchmark table
./build/voltattn --probe    # bandwidth ceiling only

# V100 box (CUDA 12.x):
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=70 && cmake --build build
./build/voltattn --bench
```

Benchmark hygiene (inherited from membound): the KV working set is rotated
across enough copies to exceed 256 MB so L2 cannot serve the stream (48 MB on
Ada makes this mandatory; V100's 6 MB L2 makes it nearly automatic), the
ceiling is re-probed on the benchmarking device rather than taken from the
spec sheet (which also absorbs HBM2 ECC overhead), and reported bytes count
compressed K+V payload plus scales.

## Limitations (honest)

- Single-query decode (batch 1, one new token) — no prefill path, no GQA
  grouping, no paged KV. Those are engineering extensions, not design changes.
- Symmetric quantization only; scales are fp16 per group of 64.
- `D ∈ {64, 128}`, `D` divisible by the lane tiling.

## License

MIT
