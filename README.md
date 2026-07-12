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

First-cut kernels (one warp per position, 32 lanes across D) hold ~93% MBU at
fp16 — but collapse to 25–60% MBU when quantized. Fewer bytes per position
means the fixed per-position cost (a 5-step warp reduce + the dependent
online-softmax update) dominates, and the memory system starves. On the V100
target the naive kernels never beat fp16 at all (q4 is a 20% *slowdown*) —
compression fully wasted.

The `q8+`/`q4+` kernels restructure to 8 lanes per position with 4 positions
in flight per warp: 4× the loads issued per iteration (128-bit loads for
INT8), a 3-step reduce, and four independent softmax chains for the scheduler
to interleave. Measured on the development GPU (RTX 4080 Laptop, sm_89,
421.5 GB/s measured read ceiling), H=32 heads, D=128:

| fmt | L | best ms | GB/s | MBU | tok/s | vs f16 |
|-----|-------|---------|-------|-------|-------|--------|
| f16 | 16384 | 0.688 | 390.1 | 92.6% | 1453 | 1.00× |
| f16+ | 16384 | 0.644 | 416.8 | 98.9% | 1553 | 1.07× |
| q8 (naive) | 16384 | 0.550 | 251.7 | 59.7% | 1819 | 1.25× |
| **q8+** | 16384 | **0.339** | **408.4** | **96.9%** | **2950** | **2.03×** |
| q4 (naive) | 16384 | 0.597 | 119.4 | 28.3% | 1675 | 1.15× |
| **q4+** | 16384 | **0.451** | 158.3 | 37.5% | 2219 | 1.53× |

At L=4096: f16 5756 tok/s → q8+ 10973 (1.91×) → q4+ 14147 (2.46×).
At L=65536: q8+ holds **98.6% MBU (2.03×)** — INT8's byte ratio is 1.94×, so
INT8 compression converts to speed at effectively full efficiency.

**f16+** — the same restructure with *no* quantization, added as the
attribution control — barely moves on Ada (1.04–1.07×): this baseline is
already at the roof, so here the speedup really is compression. On Volta it
is a different story (below).

Laptop honesty note: the memory-bound rows (f16 / f16+ / q8+) reproduce to
three digits across runs; the ALU-dense q4 rows swing with the host's DVFS
and desktop load (q4+ at L=16K measured from 0.28 ms / 2.4× on a quiet run
to 0.45 ms / 1.5× — the table shows a typical run, not the best one), and
L=65536 laptop rows vary up to ~±35% for the same reason. The V100 table
below is the stable reference.

Both kernel generations stay in the binary — the delta *is* the point.

**Remaining headroom, stated honestly:** with f16+ proving the memory path
can run at 97–99% MBU on both devices, the residual gap in q8+/q4+ is the
dequant chain itself — unpack, convert, and per-position scale loads
competing with the softmax update for issue slots. The next levers are wider
per-lane tiles and staging scales for a tile of positions. The harness
re-probes the ceiling per device; the V100 table below came from the same
`--bench` binary.

## Measured on the target: V100

Rented V100 box (Tesla V100-FHHL-16GB, sm_70, 80 SMs, CUDA 12.4 toolchain),
measured read ceiling **809 GB/s**. All 36 selftest cases pass on real
Volta. Same harness, H=32, D=128; two independent rentals — the L=4096 and
L=16384 rows reproduce within ~1% across them, the naive kernels move up to
~20% at L=65536 (ranges given below):

| fmt | L | best ms | GB/s | MBU | tok/s | vs f16 |
|-----|-------|---------|-------|-------|-------|--------|
| f16 | 16384 | 0.598 | 448.9 | 55.5% | 1672 | 1.00× |
| **f16+** | 16384 | **0.343** | **782.5** | **96.7%** | 2915 | **1.74×** |
| q8 (naive) | 16384 | 0.606 | 228.3 | 28.2% | 1650 | 0.99× |
| **q8+** | 16384 | **0.255** | 542.8 | 67.1% | **3922** | **2.35×** |
| q4 (naive) | 16384 | 0.750 | 95.1 | 11.8% | 1334 | 0.80× |
| **q4+** | 16384 | **0.254** | 280.8 | 34.7% | **3938** | **2.35×** |

At L=4096: f16 6300 tok/s → f16+ 10389 (1.65×) → q8+ 13378 (2.12×) → q4+
13563 (2.15×). At L=65536: f16+ holds **99.1% MBU**; q8+ reaches 2.41–2.71×
and q4+ **2.75–3.03×** across the two rentals.

With the f16+ control in place, the V100 story separates cleanly:

1. **The obvious fp16 kernel is latency-limited on Volta.** One warp per
   position issues too few loads in flight; it stalls at ~55% MBU where the
   identical code holds ~93% on Ada. The restructure alone — no quantization
   anywhere — is worth **1.65–1.74×** (f16+ at 97–99% MBU: effectively the
   bandwidth roof of the device).
2. **Compression converts to speed only after the kernel is restructured.**
   Against the honest f16+ baseline, INT8 buys a further **1.3–1.6×** of its
   1.94× byte ceiling, INT4 1.3–1.7× of 3.76× — the dequant chain and scale
   loads eat the rest at V100's ~18 FLOP-per-byte budget. That ALU gap, not
   the memory system, is the measured remaining headroom.
3. **Naive quantization without the restructure is worse than doing
   nothing**: q8 0.99×, q4 0.80× at L ≤ 16K — reproduced exactly across both
   rentals; at L=64K the naive kernels wobble between 0.81× and parity, never
   ahead. Shrinking bytes per position while keeping the per-position cost
   fixed starves the memory system — compression fully wasted.

(Card caveat: FHHL is the 150 W single-slot V100; SXM2 modules run 250–300 W
with higher clocks, which should help the still-ALU-bound q4+ most.)

## Correctness (gated, not asserted)

`voltattn` (no arguments) runs 36 cases — 6 formats × 6 shapes, including
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
