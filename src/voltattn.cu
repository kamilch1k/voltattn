// voltattn — fused dequant + attention decode kernel, written for sm_70.
//
// The production problem: a quantized KV cache is a *memory* win. During
// batch-1 decode, attention over the KV cache is bandwidth-bound, so the only
// way the memory win becomes a *speed* win is a kernel that streams the
// compressed bytes and dequantizes in-register — never materializing the
// fp16 cache in global memory. That fused kernel is what this repo builds:
//
//   score_j = q · dequant(K_j) / sqrt(D)   (online softmax, single pass)
//   out     = sum_j softmax_j * dequant(V_j)
//
// INT8 halves the KV bytes of fp16; INT4 quarters them. If the fused kernel
// holds the same MBU as the fp16 baseline, tokens/s scales with compression.
// The fp16 baseline runs in the same binary and each quantized kernel reports
// its measured speedup over it — the same claim structure as membound
// (github.com/kamilch1k/membound), applied to the attention path.
//
// Volta constraints honored by construction (sm_70: no FP8, no bf16, no
// cp.async, no TMA, first-gen tensor cores unused — load-dominated kernel):
//   * plain coalesced 16..64-bit per-lane loads, warp-parallel latency hiding
//   * fp32 accumulate, __expf, __shfl_xor_sync — all Volta-native
//   * dequant ALU kept lean: the compute-to-bandwidth ratio on V100
//     (~18 FLOP/B) is 4x tighter than Ada (~70 FLOP/B), so unpack is bit-ops
//     + one convert + one FMA per element, scales staged once per lane.
//
// Quantization formats (deliberately standard-shaped, clean-room):
//   q8: symmetric per-group scale, group = 64 dims along D. 1 B/elem + fp16 scale
//   q4: symmetric per-group scale, group = 64 dims along D. q in [-7,7]
//       stored as (q+8) in a nibble, packed 2/byte. 0.5 B/elem + fp16 scale
//
// Correctness: double-precision CPU reference computed on the *dequantized*
// values (q*s exactly), so the gate verifies kernel arithmetic and indexing;
// quantization error itself is a modeling choice, not a bug.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <vector>
#include <random>
#include <string>
#include <algorithm>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                \
                         cudaGetErrorName(err__), __FILE__, __LINE__,           \
                         cudaGetErrorString(err__));                            \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

constexpr int GROUP   = 64;  // quant group size along D
constexpr int WARPS   = 4;   // warps per block
constexpr int MAX_D   = 128; // shared-memory sizing

// ------------------------------------------------------------ ceiling ----
// Pure-read kernel: the empirical bandwidth ceiling this device can serve.
// MBU is reported against this measured number, not the spec sheet — that
// also absorbs ECC overhead on HBM2 parts (V100) automatically.

__global__ void read_kernel(const float4* __restrict__ src, float* __restrict__ sink, size_t n4) {
    size_t i      = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.f;
    for (; i < n4; i += stride) {
        float4 v = src[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == -1.2345678f) sink[0] = acc;
}

// -------------------------------------------------------- attn kernels ----
// One block = (32, WARPS). Grid = (H, S): head h, split s over the KV length.
// Each warp owns positions j0+w, j0+w+WARPS, ... within the block's chunk.
// Lane l owns dims [l*P, l*P+P). All P dims of a lane fall in one quant
// group (P divides GROUP), so one scale per lane per position.
//
// Online softmax state per warp: running max m, denominator l, and the
// output accumulator o[P] per lane — rescaled by exp(m_old - m_new) at each
// new max, exactly the Milakov–Gimelshein update. Block then merges its
// WARPS partials in shared memory and writes one (m, l, o[D]) partial per
// (h, s) to the workspace; a small combine kernel merges the S partials.

// workspace layout: [H*S] slots of (2 + D) floats:
//   ws[slot*(D+2) + 0] = m,  + 1 = l,  + 2 + d = o[d]

template <int P>
__device__ inline void merge_and_store(float m, float l, const float* o,
                                       float* ws, int D, int slot) {
    __shared__ float shM[WARPS];
    __shared__ float shL[WARPS];
    __shared__ float shO[WARPS][MAX_D];
    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    if (lane == 0) { shM[w] = m; shL[w] = l; }
    #pragma unroll
    for (int p = 0; p < P; ++p) shO[w][lane * P + p] = o[p];
    __syncthreads();

    const int t = w * 32 + lane; // 0..127
    float mStar = -FLT_MAX;
    for (int i = 0; i < WARPS; ++i) mStar = fmaxf(mStar, shM[i]);
    float lStar = 0.f;
    for (int i = 0; i < WARPS; ++i) lStar += shL[i] * __expf(shM[i] - mStar);
    float od = 0.f;
    if (t < D)
        for (int i = 0; i < WARPS; ++i) od += shO[i][t] * __expf(shM[i] - mStar);

    float* slotp = ws + (size_t)slot * (D + 2);
    if (t == 0) { slotp[0] = mStar; slotp[1] = lStar; }
    if (t < D) slotp[2 + t] = od;
}

// fp16 baseline: K,V are __half[H][L][D]
template <int P>
__global__ void attn_f16(const __half* __restrict__ K, const __half* __restrict__ V,
                         const __half* __restrict__ q, float* __restrict__ ws,
                         int H, int L, int S, int chunk) {
    const int D    = P * 32;
    const int h    = blockIdx.x;
    const int s    = blockIdx.y;
    const int lane = threadIdx.x;
    const int j0   = s * chunk;
    const int j1   = min(L, j0 + chunk);

    float qr[P];
    {
        const float scale = rsqrtf((float)D);
        #pragma unroll
        for (int p = 0; p < P; ++p)
            qr[p] = __half2float(q[(size_t)h * D + lane * P + p]) * scale;
    }

    float m = -FLT_MAX, l = 0.f, o[P];
    #pragma unroll
    for (int p = 0; p < P; ++p) o[p] = 0.f;

    for (int j = j0 + threadIdx.y; j < j1; j += WARPS) {
        const size_t row = ((size_t)h * L + j) * D + (size_t)lane * P;
        float kf[P], vf[P];
        if constexpr (P == 4) {
            const float2 kw = *reinterpret_cast<const float2*>(K + row);
            const float2 vw = *reinterpret_cast<const float2*>(V + row);
            const __half2* kh = reinterpret_cast<const __half2*>(&kw);
            const __half2* vh = reinterpret_cast<const __half2*>(&vw);
            #pragma unroll
            for (int k = 0; k < 2; ++k) {
                const float2 a = __half22float2(kh[k]);
                const float2 b = __half22float2(vh[k]);
                kf[2 * k] = a.x; kf[2 * k + 1] = a.y;
                vf[2 * k] = b.x; vf[2 * k + 1] = b.y;
            }
        } else {
            const float kw = *reinterpret_cast<const float*>(K + row);
            const float vw = *reinterpret_cast<const float*>(V + row);
            const __half2 kh = *reinterpret_cast<const __half2*>(&kw);
            const __half2 vh = *reinterpret_cast<const __half2*>(&vw);
            const float2 a = __half22float2(kh);
            const float2 b = __half22float2(vh);
            kf[0] = a.x; kf[1] = a.y;
            vf[0] = b.x; vf[1] = b.y;
        }
        float acc = 0.f;
        #pragma unroll
        for (int p = 0; p < P; ++p) acc = fmaf(qr[p], kf[p], acc);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_xor_sync(0xffffffffu, acc, off);

        const float mNew = fmaxf(m, acc);
        const float c    = __expf(m - mNew);
        const float e    = __expf(acc - mNew);
        l = l * c + e;
        m = mNew;
        #pragma unroll
        for (int p = 0; p < P; ++p) o[p] = o[p] * c + e * vf[p];
    }
    merge_and_store<P>(m, l, o, ws, D, blockIdx.x * S + s);
}

// q8: K,V int8 [H][L][D] + fp16 per-group scales [H][L][D/GROUP]
template <int P>
__global__ void attn_q8(const int8_t* __restrict__ K, const int8_t* __restrict__ V,
                        const __half* __restrict__ scK, const __half* __restrict__ scV,
                        const __half* __restrict__ q, float* __restrict__ ws,
                        int H, int L, int S, int chunk) {
    const int D    = P * 32;
    const int G    = D / GROUP;
    const int h    = blockIdx.x;
    const int s    = blockIdx.y;
    const int lane = threadIdx.x;
    const int g    = (lane * P) / GROUP;
    const int j0   = s * chunk;
    const int j1   = min(L, j0 + chunk);

    float qr[P];
    {
        const float scale = rsqrtf((float)D);
        #pragma unroll
        for (int p = 0; p < P; ++p)
            qr[p] = __half2float(q[(size_t)h * D + lane * P + p]) * scale;
    }

    float m = -FLT_MAX, l = 0.f, o[P];
    #pragma unroll
    for (int p = 0; p < P; ++p) o[p] = 0.f;

    for (int j = j0 + threadIdx.y; j < j1; j += WARPS) {
        const size_t row  = ((size_t)h * L + j) * D + (size_t)lane * P;
        const size_t srow = ((size_t)h * L + j) * G + g;
        int kq[P], vq[P];
        if constexpr (P == 4) {
            const char4 kc = *reinterpret_cast<const char4*>(K + row);
            const char4 vc = *reinterpret_cast<const char4*>(V + row);
            kq[0] = kc.x; kq[1] = kc.y; kq[2] = kc.z; kq[3] = kc.w;
            vq[0] = vc.x; vq[1] = vc.y; vq[2] = vc.z; vq[3] = vc.w;
        } else {
            const char2 kc = *reinterpret_cast<const char2*>(K + row);
            const char2 vc = *reinterpret_cast<const char2*>(V + row);
            kq[0] = kc.x; kq[1] = kc.y;
            vq[0] = vc.x; vq[1] = vc.y;
        }
        const float ks = __half2float(scK[srow]);
        const float vs = __half2float(scV[srow]);

        float acc = 0.f;
        #pragma unroll
        for (int p = 0; p < P; ++p) acc = fmaf(qr[p], (float)kq[p], acc);
        acc *= ks; // all P dims of this lane share one group scale
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_xor_sync(0xffffffffu, acc, off);

        const float mNew = fmaxf(m, acc);
        const float c    = __expf(m - mNew);
        const float e    = __expf(acc - mNew);
        l = l * c + e;
        m = mNew;
        const float ev = e * vs;
        #pragma unroll
        for (int p = 0; p < P; ++p) o[p] = o[p] * c + ev * (float)vq[p];
    }
    merge_and_store<P>(m, l, o, ws, D, blockIdx.x * S + s);
}

// q4: K,V packed nibbles [H][L][D/2] + fp16 per-group scales.
// byte k of a position row holds dims 2k (lo) and 2k+1 (hi), value = nibble-8.
template <int P>
__global__ void attn_q4(const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
                        const __half* __restrict__ scK, const __half* __restrict__ scV,
                        const __half* __restrict__ q, float* __restrict__ ws,
                        int H, int L, int S, int chunk) {
    const int D    = P * 32;
    const int G    = D / GROUP;
    const int h    = blockIdx.x;
    const int s    = blockIdx.y;
    const int lane = threadIdx.x;
    const int g    = (lane * P) / GROUP;
    const int j0   = s * chunk;
    const int j1   = min(L, j0 + chunk);

    float qr[P];
    {
        const float scale = rsqrtf((float)D);
        #pragma unroll
        for (int p = 0; p < P; ++p)
            qr[p] = __half2float(q[(size_t)h * D + lane * P + p]) * scale;
    }

    float m = -FLT_MAX, l = 0.f, o[P];
    #pragma unroll
    for (int p = 0; p < P; ++p) o[p] = 0.f;

    for (int j = j0 + threadIdx.y; j < j1; j += WARPS) {
        const size_t row  = ((size_t)h * L + j) * (D / 2) + (size_t)lane * (P / 2);
        const size_t srow = ((size_t)h * L + j) * G + g;
        int kq[P], vq[P];
        if constexpr (P == 4) {
            const uint16_t kb = *reinterpret_cast<const uint16_t*>(K + row);
            const uint16_t vb = *reinterpret_cast<const uint16_t*>(V + row);
            kq[0] = (kb & 0xF) - 8; kq[1] = ((kb >> 4) & 0xF) - 8;
            kq[2] = ((kb >> 8) & 0xF) - 8; kq[3] = ((kb >> 12) & 0xF) - 8;
            vq[0] = (vb & 0xF) - 8; vq[1] = ((vb >> 4) & 0xF) - 8;
            vq[2] = ((vb >> 8) & 0xF) - 8; vq[3] = ((vb >> 12) & 0xF) - 8;
        } else {
            const uint8_t kb = K[row];
            const uint8_t vb = V[row];
            kq[0] = (kb & 0xF) - 8; kq[1] = (kb >> 4) - 8;
            vq[0] = (vb & 0xF) - 8; vq[1] = (vb >> 4) - 8;
        }
        const float ks = __half2float(scK[srow]);
        const float vs = __half2float(scV[srow]);

        float acc = 0.f;
        #pragma unroll
        for (int p = 0; p < P; ++p) acc = fmaf(qr[p], (float)kq[p], acc);
        acc *= ks;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_xor_sync(0xffffffffu, acc, off);

        const float mNew = fmaxf(m, acc);
        const float c    = __expf(m - mNew);
        const float e    = __expf(acc - mNew);
        l = l * c + e;
        m = mNew;
        const float ev = e * vs;
        #pragma unroll
        for (int p = 0; p < P; ++p) o[p] = o[p] * c + ev * (float)vq[p];
    }
    merge_and_store<P>(m, l, o, ws, D, blockIdx.x * S + s);
}

// ----------------------------------------- f16+/q8+/q4+ (sub-warp MLP) ----
// First-cut q8/q4 above hold only ~25-60% MBU: with 2-4x fewer bytes per
// position, the fixed per-position cost (a 32-lane shuffle reduce + the
// dependent online-softmax update) dominates and the memory system starves.
// Fix: 8 lanes per position, 4 positions in flight per warp — 4x the loads
// issued per iteration (128-bit loads for q8), a 3-shuffle reduce instead of
// 5, and four independent softmax chains the scheduler can interleave.
//
// f16+ is the identical restructure with no quantization. It exists to split
// the attribution: q8+ over f16+ is compression converting to speed at equal
// memory-level parallelism; f16+ over f16 is parallelism the baseline left
// on the table (on Volta: a lot — the baseline itself is latency-limited).

template <int Pl> // dims per lane; D = 8 * Pl
__device__ inline void merge_and_store16(float m, float l, const float* o,
                                         float* ws, int D, int slot) {
    constexpr int NPART = WARPS * 4;
    __shared__ float shM[NPART];
    __shared__ float shL[NPART];
    __shared__ float shO[NPART][MAX_D];
    const int lane  = threadIdx.x;
    const int sl    = lane & 7;
    const int subId = threadIdx.y * 4 + (lane >> 3);
    if (sl == 0) { shM[subId] = m; shL[subId] = l; }
    #pragma unroll
    for (int p = 0; p < Pl; ++p) shO[subId][sl * Pl + p] = o[p];
    __syncthreads();

    const int t = threadIdx.y * 32 + lane;
    float mStar = -FLT_MAX;
    for (int i = 0; i < NPART; ++i) mStar = fmaxf(mStar, shM[i]);
    float lStar = 0.f;
    for (int i = 0; i < NPART; ++i) lStar += shL[i] * __expf(shM[i] - mStar);
    float od = 0.f;
    if (t < D)
        for (int i = 0; i < NPART; ++i) od += shO[i][t] * __expf(shM[i] - mStar);

    float* slotp = ws + (size_t)slot * (D + 2);
    if (t == 0) { slotp[0] = mStar; slotp[1] = lStar; }
    if (t < D) slotp[2 + t] = od;
}

template <int Pl>
__global__ void attn_f16_mlp(const __half* __restrict__ K, const __half* __restrict__ V,
                             const __half* __restrict__ q, float* __restrict__ ws,
                             int H, int L, int S, int chunk) {
    const int D    = Pl * 8;
    const int h    = blockIdx.x;
    const int s    = blockIdx.y;
    const int lane = threadIdx.x;
    const int sl   = lane & 7;
    const int sub  = threadIdx.y * 4 + (lane >> 3);
    const unsigned subMask = 0xFFu << (lane & 24); // my 8-lane group
    const int j0   = s * chunk;
    const int j1   = min(L, j0 + chunk);

    float qr[Pl];
    {
        const float scale = rsqrtf((float)D);
        #pragma unroll
        for (int p = 0; p < Pl; ++p)
            qr[p] = __half2float(q[(size_t)h * D + sl * Pl + p]) * scale;
    }

    float m = -FLT_MAX, l = 0.f, o[Pl];
    #pragma unroll
    for (int p = 0; p < Pl; ++p) o[p] = 0.f;

    for (int j = j0 + sub; j < j1; j += WARPS * 4) {
        const size_t row = ((size_t)h * L + j) * D + (size_t)sl * Pl;
        float kf[Pl], vf[Pl];
        #pragma unroll
        for (int seg = 0; seg < Pl / 8; ++seg) { // 8 halves = one 128-bit load
            const int4 kw = *reinterpret_cast<const int4*>(K + row + seg * 8);
            const int4 vw = *reinterpret_cast<const int4*>(V + row + seg * 8);
            const __half2* kh = reinterpret_cast<const __half2*>(&kw);
            const __half2* vh = reinterpret_cast<const __half2*>(&vw);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float2 a = __half22float2(kh[k]);
                const float2 b = __half22float2(vh[k]);
                kf[seg * 8 + 2 * k] = a.x; kf[seg * 8 + 2 * k + 1] = a.y;
                vf[seg * 8 + 2 * k] = b.x; vf[seg * 8 + 2 * k + 1] = b.y;
            }
        }
        float acc = 0.f;
        #pragma unroll
        for (int p = 0; p < Pl; ++p) acc = fmaf(qr[p], kf[p], acc);
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1)
            acc += __shfl_xor_sync(subMask, acc, off);

        const float mNew = fmaxf(m, acc);
        const float c    = __expf(m - mNew);
        const float e    = __expf(acc - mNew);
        l = l * c + e;
        m = mNew;
        #pragma unroll
        for (int p = 0; p < Pl; ++p) o[p] = o[p] * c + e * vf[p];
    }
    merge_and_store16<Pl>(m, l, o, ws, D, blockIdx.x * S + s);
}

template <int Pl>
__global__ void attn_q8_mlp(const int8_t* __restrict__ K, const int8_t* __restrict__ V,
                            const __half* __restrict__ scK, const __half* __restrict__ scV,
                            const __half* __restrict__ q, float* __restrict__ ws,
                            int H, int L, int S, int chunk) {
    const int D    = Pl * 8;
    const int G    = D / GROUP;
    const int h    = blockIdx.x;
    const int s    = blockIdx.y;
    const int lane = threadIdx.x;
    const int sl   = lane & 7;
    const int sub  = threadIdx.y * 4 + (lane >> 3);
    const unsigned subMask = 0xFFu << (lane & 24); // my 8-lane group
    const int g    = (sl * Pl) / GROUP;
    const int j0   = s * chunk;
    const int j1   = min(L, j0 + chunk);

    float qr[Pl];
    {
        const float scale = rsqrtf((float)D);
        #pragma unroll
        for (int p = 0; p < Pl; ++p)
            qr[p] = __half2float(q[(size_t)h * D + sl * Pl + p]) * scale;
    }

    float m = -FLT_MAX, l = 0.f, o[Pl];
    #pragma unroll
    for (int p = 0; p < Pl; ++p) o[p] = 0.f;

    for (int j = j0 + sub; j < j1; j += WARPS * 4) {
        const size_t row  = ((size_t)h * L + j) * D + (size_t)sl * Pl;
        const size_t srow = ((size_t)h * L + j) * G + g;
        int4 kc, vc; // Pl=16: one 128-bit load. Pl=8: 64-bit via int2.
        if constexpr (Pl == 16) {
            kc = *reinterpret_cast<const int4*>(K + row);
            vc = *reinterpret_cast<const int4*>(V + row);
        } else {
            const int2 k2 = *reinterpret_cast<const int2*>(K + row);
            const int2 v2 = *reinterpret_cast<const int2*>(V + row);
            kc = make_int4(k2.x, k2.y, 0, 0);
            vc = make_int4(v2.x, v2.y, 0, 0);
        }
        const int8_t* kb = reinterpret_cast<const int8_t*>(&kc);
        const int8_t* vb = reinterpret_cast<const int8_t*>(&vc);
        const float ks = __half2float(scK[srow]);
        const float vs = __half2float(scV[srow]);

        float acc = 0.f;
        #pragma unroll
        for (int p = 0; p < Pl; ++p) acc = fmaf(qr[p], (float)kb[p], acc);
        acc *= ks;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1)
            acc += __shfl_xor_sync(subMask, acc, off);

        const float mNew = fmaxf(m, acc);
        const float c    = __expf(m - mNew);
        const float e    = __expf(acc - mNew);
        l = l * c + e;
        m = mNew;
        const float ev = e * vs;
        #pragma unroll
        for (int p = 0; p < Pl; ++p) o[p] = o[p] * c + ev * (float)vb[p];
    }
    merge_and_store16<Pl>(m, l, o, ws, D, blockIdx.x * S + s);
}

template <int Pl>
__global__ void attn_q4_mlp(const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
                            const __half* __restrict__ scK, const __half* __restrict__ scV,
                            const __half* __restrict__ q, float* __restrict__ ws,
                            int H, int L, int S, int chunk) {
    const int D    = Pl * 8;
    const int G    = D / GROUP;
    const int h    = blockIdx.x;
    const int s    = blockIdx.y;
    const int lane = threadIdx.x;
    const int sl   = lane & 7;
    const int sub  = threadIdx.y * 4 + (lane >> 3);
    const unsigned subMask = 0xFFu << (lane & 24); // my 8-lane group
    const int g    = (sl * Pl) / GROUP;
    const int j0   = s * chunk;
    const int j1   = min(L, j0 + chunk);

    float qr[Pl];
    {
        const float scale = rsqrtf((float)D);
        #pragma unroll
        for (int p = 0; p < Pl; ++p)
            qr[p] = __half2float(q[(size_t)h * D + sl * Pl + p]) * scale;
    }

    float m = -FLT_MAX, l = 0.f, o[Pl];
    #pragma unroll
    for (int p = 0; p < Pl; ++p) o[p] = 0.f;

    for (int j = j0 + sub; j < j1; j += WARPS * 4) {
        const size_t row  = ((size_t)h * L + j) * (D / 2) + (size_t)sl * (Pl / 2);
        const size_t srow = ((size_t)h * L + j) * G + g;
        uint2 kc, vc; // Pl=16: 8 bytes = 16 nibbles. Pl=8: 4 bytes.
        if constexpr (Pl == 16) {
            kc = *reinterpret_cast<const uint2*>(K + row);
            vc = *reinterpret_cast<const uint2*>(V + row);
        } else {
            kc = make_uint2(*reinterpret_cast<const uint32_t*>(K + row), 0u);
            vc = make_uint2(*reinterpret_cast<const uint32_t*>(V + row), 0u);
        }
        const uint8_t* kb = reinterpret_cast<const uint8_t*>(&kc);
        const uint8_t* vb = reinterpret_cast<const uint8_t*>(&vc);
        const float ks = __half2float(scK[srow]);
        const float vs = __half2float(scV[srow]);

        float acc = 0.f;
        #pragma unroll
        for (int k = 0; k < Pl / 2; ++k) {
            acc = fmaf(qr[2 * k],     (float)((int)(kb[k] & 0xF) - 8), acc);
            acc = fmaf(qr[2 * k + 1], (float)((int)(kb[k] >> 4)  - 8), acc);
        }
        acc *= ks;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1)
            acc += __shfl_xor_sync(subMask, acc, off);

        const float mNew = fmaxf(m, acc);
        const float c    = __expf(m - mNew);
        const float e    = __expf(acc - mNew);
        l = l * c + e;
        m = mNew;
        const float ev = e * vs;
        #pragma unroll
        for (int k = 0; k < Pl / 2; ++k) {
            o[2 * k]     = o[2 * k]     * c + ev * (float)((int)(vb[k] & 0xF) - 8);
            o[2 * k + 1] = o[2 * k + 1] * c + ev * (float)((int)(vb[k] >> 4)  - 8);
        }
    }
    merge_and_store16<Pl>(m, l, o, ws, D, blockIdx.x * S + s);
}

// Merge the S per-split partials of each head: same log-sum-exp merge.
__global__ void combine(const float* __restrict__ ws, __half* __restrict__ out,
                        int S, int D) {
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    float mStar = -FLT_MAX;
    for (int s = 0; s < S; ++s)
        mStar = fmaxf(mStar, ws[(size_t)(h * S + s) * (D + 2)]);
    float lStar = 0.f;
    for (int s = 0; s < S; ++s) {
        const float* slot = ws + (size_t)(h * S + s) * (D + 2);
        lStar += slot[1] * __expf(slot[0] - mStar);
    }
    if (t < D) {
        float od = 0.f;
        for (int s = 0; s < S; ++s) {
            const float* slot = ws + (size_t)(h * S + s) * (D + 2);
            od += slot[2 + t] * __expf(slot[0] - mStar);
        }
        out[(size_t)h * D + t] = __float2half(od / lStar);
    }
}

// ------------------------------------------------------------- host -------

enum class Fmt { F16, F16M, Q8, Q8M, Q4, Q4M };
static const char* fmt_name(Fmt f) {
    switch (f) {
        case Fmt::F16:  return "f16";
        case Fmt::F16M: return "f16+";
        case Fmt::Q8:   return "q8 ";
        case Fmt::Q8M:  return "q8+";
        case Fmt::Q4:   return "q4 ";
        default:        return "q4+";
    }
}
static bool is_q4(Fmt f) { return f == Fmt::Q4 || f == Fmt::Q4M; }
static bool is_f16(Fmt f) { return f == Fmt::F16 || f == Fmt::F16M; }

struct Quantized {
    std::vector<int8_t>  q8;      // [H*L*D]
    std::vector<uint8_t> q4;      // [H*L*D/2]
    std::vector<__half>  scale;   // [H*L*D/GROUP]
};

// Symmetric per-group quantizer. q8: round(x/s), s = max|x|/127.
// q4: round(x/s) clamped to [-7,7], s = max|x|/7, stored as nibble+8.
static Quantized quantize(const std::vector<float>& x, int H, int L, int D, bool four_bit) {
    const int G = D / GROUP;
    Quantized r;
    r.scale.resize((size_t)H * L * G);
    if (four_bit) r.q4.resize((size_t)H * L * D / 2);
    else          r.q8.resize((size_t)H * L * D);
    for (size_t rowg = 0; rowg < (size_t)H * L * G; ++rowg) {
        const size_t base = (rowg / G) * D + (rowg % G) * GROUP;
        float amax = 0.f;
        for (int i = 0; i < GROUP; ++i) amax = std::max(amax, std::fabs(x[base + i]));
        const float s = amax > 0.f ? amax / (four_bit ? 7.f : 127.f) : 1.f;
        r.scale[rowg] = __float2half(s);
        const float sh = __half2float(r.scale[rowg]); // quantize against the stored scale
        for (int i = 0; i < GROUP; ++i) {
            const int lim = four_bit ? 7 : 127;
            int v = (int)std::lrint(x[base + i] / sh);
            v = std::max(-lim, std::min(lim, v));
            if (four_bit) {
                uint8_t& b = r.q4[(base + i) / 2];
                const uint8_t nib = (uint8_t)(v + 8);
                if ((base + i) % 2 == 0) b = (uint8_t)((b & 0xF0) | nib);
                else                     b = (uint8_t)((b & 0x0F) | (nib << 4));
            } else {
                r.q8[base + i] = (int8_t)v;
            }
        }
    }
    return r;
}

// Dequantized value as the kernel computes it, in double.
static double dq(const Quantized& z, int H, int L, int D, bool four_bit, size_t idx) {
    (void)H; (void)L;
    const int G = D / GROUP;
    const size_t rowg = (idx / D) * G + (idx % D) / GROUP;
    const double s = (double)__half2float(z.scale[rowg]);
    int v;
    if (four_bit) {
        const uint8_t b = z.q4[idx / 2];
        v = (int)((idx % 2 == 0) ? (b & 0xF) : (b >> 4)) - 8;
    } else {
        v = z.q8[idx];
    }
    return s * v;
}

// Double-precision reference attention over the dequantized cache.
static std::vector<float> reference(const std::vector<float>& kd, const std::vector<float>& vd,
                                    const std::vector<__half>& q, int H, int L, int D) {
    std::vector<float> out((size_t)H * D);
    std::vector<double> sc(L), od(D);
    for (int h = 0; h < H; ++h) {
        double mx = -1e300;
        for (int j = 0; j < L; ++j) {
            double acc = 0.0;
            for (int d = 0; d < D; ++d)
                acc += (double)__half2float(q[(size_t)h * D + d]) * kd[((size_t)h * L + j) * D + d];
            sc[j] = acc / std::sqrt((double)D);
            mx = std::max(mx, sc[j]);
        }
        double Z = 0.0;
        std::fill(od.begin(), od.end(), 0.0);
        for (int j = 0; j < L; ++j) {
            const double p = std::exp(sc[j] - mx);
            Z += p;
            for (int d = 0; d < D; ++d)
                od[d] += p * vd[((size_t)h * L + j) * D + d];
        }
        for (int d = 0; d < D; ++d) out[(size_t)h * D + d] = (float)(od[d] / Z);
    }
    return out;
}

struct DeviceBufs {
    void *K = nullptr, *V = nullptr, *scK = nullptr, *scV = nullptr;
    __half* q  = nullptr;
    __half* out = nullptr;
    float*  ws  = nullptr;
};

static void launch(Fmt f, const DeviceBufs& b, int H, int L, int D, int S) {
    const int chunk = (L + S - 1) / S;
    const dim3 grid((unsigned)H, (unsigned)S);
    const dim3 block(32, WARPS);
    const int8_t*  K8 = (const int8_t*)b.K;
    const int8_t*  V8 = (const int8_t*)b.V;
    const uint8_t* K4 = (const uint8_t*)b.K;
    const uint8_t* V4 = (const uint8_t*)b.V;
    const __half*  sK = (const __half*)b.scK;
    const __half*  sV = (const __half*)b.scV;
    if (D == 128) {
        switch (f) {
            case Fmt::F16:  attn_f16<4><<<grid, block>>>((const __half*)b.K, (const __half*)b.V, b.q, b.ws, H, L, S, chunk); break;
            case Fmt::F16M: attn_f16_mlp<16><<<grid, block>>>((const __half*)b.K, (const __half*)b.V, b.q, b.ws, H, L, S, chunk); break;
            case Fmt::Q8:  attn_q8<4><<<grid, block>>>(K8, V8, sK, sV, b.q, b.ws, H, L, S, chunk); break;
            case Fmt::Q8M: attn_q8_mlp<16><<<grid, block>>>(K8, V8, sK, sV, b.q, b.ws, H, L, S, chunk); break;
            case Fmt::Q4:  attn_q4<4><<<grid, block>>>(K4, V4, sK, sV, b.q, b.ws, H, L, S, chunk); break;
            default:       attn_q4_mlp<16><<<grid, block>>>(K4, V4, sK, sV, b.q, b.ws, H, L, S, chunk); break;
        }
    } else {
        switch (f) {
            case Fmt::F16:  attn_f16<2><<<grid, block>>>((const __half*)b.K, (const __half*)b.V, b.q, b.ws, H, L, S, chunk); break;
            case Fmt::F16M: attn_f16_mlp<8><<<grid, block>>>((const __half*)b.K, (const __half*)b.V, b.q, b.ws, H, L, S, chunk); break;
            case Fmt::Q8:  attn_q8<2><<<grid, block>>>(K8, V8, sK, sV, b.q, b.ws, H, L, S, chunk); break;
            case Fmt::Q8M: attn_q8_mlp<8><<<grid, block>>>(K8, V8, sK, sV, b.q, b.ws, H, L, S, chunk); break;
            case Fmt::Q4:  attn_q4<2><<<grid, block>>>(K4, V4, sK, sV, b.q, b.ws, H, L, S, chunk); break;
            default:       attn_q4_mlp<8><<<grid, block>>>(K4, V4, sK, sV, b.q, b.ws, H, L, S, chunk); break;
        }
    }
    combine<<<H, 128>>>(b.ws, b.out, S, D);
}

// One correctness case: random data -> quantize -> kernel vs double reference.
static bool run_case(Fmt f, int H, int L, int D, int S, unsigned seed, double* maxerr_out) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    const size_t n = (size_t)H * L * D;
    std::vector<float> kx(n), vx(n);
    for (auto& v : kx) v = nd(rng);
    for (auto& v : vx) v = nd(rng);
    std::vector<__half> q((size_t)H * D);
    for (auto& v : q) v = __float2half(nd(rng));

    // exact dequantized copies for the reference
    std::vector<float> kd(n), vd(n);
    Quantized zk, zv;
    if (is_f16(f)) {
        for (size_t i = 0; i < n; ++i) kd[i] = __half2float(__float2half(kx[i]));
        for (size_t i = 0; i < n; ++i) vd[i] = __half2float(__float2half(vx[i]));
    } else {
        const bool fb = is_q4(f);
        zk = quantize(kx, H, L, D, fb);
        zv = quantize(vx, H, L, D, fb);
        for (size_t i = 0; i < n; ++i) kd[i] = (float)dq(zk, H, L, D, fb, i);
        for (size_t i = 0; i < n; ++i) vd[i] = (float)dq(zv, H, L, D, fb, i);
    }
    const std::vector<float> ref = reference(kd, vd, q, H, L, D);

    DeviceBufs b;
    const int G = D / GROUP;
    if (is_f16(f)) {
        std::vector<__half> kh(n), vh(n);
        for (size_t i = 0; i < n; ++i) kh[i] = __float2half(kx[i]);
        for (size_t i = 0; i < n; ++i) vh[i] = __float2half(vx[i]);
        CUDA_CHECK(cudaMalloc(&b.K, n * 2)); CUDA_CHECK(cudaMalloc(&b.V, n * 2));
        CUDA_CHECK(cudaMemcpy(b.K, kh.data(), n * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b.V, vh.data(), n * 2, cudaMemcpyHostToDevice));
    } else if (!is_q4(f)) {
        CUDA_CHECK(cudaMalloc(&b.K, n)); CUDA_CHECK(cudaMalloc(&b.V, n));
        CUDA_CHECK(cudaMemcpy(b.K, zk.q8.data(), n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b.V, zv.q8.data(), n, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMalloc(&b.K, n / 2)); CUDA_CHECK(cudaMalloc(&b.V, n / 2));
        CUDA_CHECK(cudaMemcpy(b.K, zk.q4.data(), n / 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b.V, zv.q4.data(), n / 2, cudaMemcpyHostToDevice));
    }
    if (!is_f16(f)) {
        const size_t ns = (size_t)H * L * G;
        CUDA_CHECK(cudaMalloc(&b.scK, ns * 2)); CUDA_CHECK(cudaMalloc(&b.scV, ns * 2));
        CUDA_CHECK(cudaMemcpy(b.scK, zk.scale.data(), ns * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b.scV, zv.scale.data(), ns * 2, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&b.q, (size_t)H * D * 2));
    CUDA_CHECK(cudaMemcpy(b.q, q.data(), (size_t)H * D * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&b.out, (size_t)H * D * 2));
    CUDA_CHECK(cudaMalloc(&b.ws, (size_t)H * S * (D + 2) * 4));

    launch(f, b, H, L, D, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> got((size_t)H * D);
    CUDA_CHECK(cudaMemcpy(got.data(), b.out, (size_t)H * D * 2, cudaMemcpyDeviceToHost));
    double maxerr = 0.0;
    bool ok = true;
    for (size_t i = 0; i < got.size(); ++i) {
        const double g = (double)__half2float(got[i]);
        const double r = (double)ref[i];
        const double err = std::fabs(g - r);
        maxerr = std::max(maxerr, err);
        if (err > 3e-3 + 2e-2 * std::fabs(r)) ok = false;
    }
    *maxerr_out = maxerr;
    for (void* p : {b.K, b.V, b.scK, b.scV, (void*)b.q, (void*)b.out, (void*)b.ws})
        if (p) CUDA_CHECK(cudaFree(p));
    return ok;
}

static int selftest() {
    struct Case { int H, L, D, S; };
    const Case cases[] = {
        {8, 257, 128, 1}, {8, 257, 128, 3}, {2, 4096, 128, 5},
        {40, 1000, 128, 2}, {8, 129, 64, 1}, {8, 513, 64, 4},
    };
    int fails = 0;
    for (const Fmt f : {Fmt::F16, Fmt::F16M, Fmt::Q8, Fmt::Q8M, Fmt::Q4, Fmt::Q4M}) {
        for (const Case& c : cases) {
            double maxerr = 0.0;
            const bool ok = run_case(f, c.H, c.L, c.D, c.S,
                                     0xC0FFEEu ^ (unsigned)(c.H * 131 + c.L * 7 + c.D + c.S), &maxerr);
            std::printf("%s  H=%-3d L=%-5d D=%-3d S=%-2d  maxerr=%.2e  %s\n",
                        fmt_name(f), c.H, c.L, c.D, c.S, maxerr, ok ? "PASS" : "FAIL");
            if (!ok) ++fails;
        }
    }
    if (fails) { std::printf("%d case(s) FAILED\n", fails); return 1; }
    std::printf("all cases passed\n");
    return 0;
}

// ------------------------------------------------------------- bench ------

static double probe_ceiling() {
    const size_t bytes = 256ull << 20;
    const size_t n4    = bytes / 16;
    float4* src; float* sink;
    CUDA_CHECK(cudaMalloc(&src, bytes));
    CUDA_CHECK(cudaMalloc(&sink, 4));
    CUDA_CHECK(cudaMemset(src, 1, bytes));
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    double best = 0.0;
    for (int r = 0; r < 20; ++r) {
        CUDA_CHECK(cudaEventRecord(e0));
        read_kernel<<<4096, 256>>>(src, sink, n4);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        best = std::max(best, (double)bytes / (ms * 1e6));
    }
    CUDA_CHECK(cudaFree(src)); CUDA_CHECK(cudaFree(sink));
    CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
    return best; // GB/s
}

// cheap deterministic fill in [-1, 1] — bench data, values don't matter
static void fill_fast(std::vector<float>& v, uint64_t s) {
    for (auto& x : v) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        x = (float)((int32_t)(s >> 33)) * (1.f / 2147483648.f);
    }
}

static void bench() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);
    const double ceil_gbs = probe_ceiling();
    std::printf("measured read ceiling: %.1f GB/s\n\n", ceil_gbs);

    const int H = 32, D = 128, G = D / GROUP;
    std::printf("| fmt | L      | KV bytes | best ms | GB/s   | MBU   | tok/s  | vs f16 |\n");
    std::printf("|-----|--------|----------|---------|--------|-------|--------|--------|\n");

    for (const int L : {4096, 16384, 65536}) {
        const size_t n = (size_t)H * L * D;
        std::vector<float> kx(n), vx(n);
        fill_fast(kx, 0x1234u + (unsigned)L);
        fill_fast(vx, 0x5678u + (unsigned)L);
        std::vector<__half> q((size_t)H * D);
        for (size_t i = 0; i < q.size(); ++i) q[i] = __float2half((float)((int)(i % 7) - 3) * 0.25f);

        double f16_ms = 0.0;
        for (const Fmt f : {Fmt::F16, Fmt::F16M, Fmt::Q8, Fmt::Q8M, Fmt::Q4, Fmt::Q4M}) {
            size_t payload, sbytes = 0;
            if (is_f16(f))        payload = n * 2;
            else if (!is_q4(f)) { payload = n;     sbytes = (size_t)H * L * G * 2; }
            else                { payload = n / 2; sbytes = (size_t)H * L * G * 2; }
            const size_t copy_bytes = 2 * (payload + sbytes); // K and V streams
            // rotate enough copies that the working set exceeds 256MB (cap: 16)
            const int copies = (int)std::min<size_t>(16, ((256ull << 20) + copy_bytes - 1) / copy_bytes);

            // host-side prep
            Quantized zk, zv;
            std::vector<__half> kh, vh;
            if (is_f16(f)) {
                kh.resize(n); vh.resize(n);
                for (size_t i = 0; i < n; ++i) kh[i] = __float2half(kx[i]);
                for (size_t i = 0; i < n; ++i) vh[i] = __float2half(vx[i]);
            } else {
                const bool fb = is_q4(f);
                zk = quantize(kx, H, L, D, fb);
                zv = quantize(vx, H, L, D, fb);
            }

            std::vector<DeviceBufs> bufs(copies);
            for (int r = 0; r < copies; ++r) {
                DeviceBufs& b = bufs[r];
                if (is_f16(f)) {
                    CUDA_CHECK(cudaMalloc(&b.K, payload)); CUDA_CHECK(cudaMalloc(&b.V, payload));
                    CUDA_CHECK(cudaMemcpy(b.K, kh.data(), payload, cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(b.V, vh.data(), payload, cudaMemcpyHostToDevice));
                } else {
                    const void* ksrc = !is_q4(f) ? (const void*)zk.q8.data() : (const void*)zk.q4.data();
                    const void* vsrc = !is_q4(f) ? (const void*)zv.q8.data() : (const void*)zv.q4.data();
                    CUDA_CHECK(cudaMalloc(&b.K, payload)); CUDA_CHECK(cudaMalloc(&b.V, payload));
                    CUDA_CHECK(cudaMemcpy(b.K, ksrc, payload, cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(b.V, vsrc, payload, cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMalloc(&b.scK, sbytes)); CUDA_CHECK(cudaMalloc(&b.scV, sbytes));
                    CUDA_CHECK(cudaMemcpy(b.scK, zk.scale.data(), sbytes, cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(b.scV, zv.scale.data(), sbytes, cudaMemcpyHostToDevice));
                }
            }
            const int S = std::max(1, std::min(prop.multiProcessorCount * 2 / H, (L + 511) / 512));
            DeviceBufs shared;
            CUDA_CHECK(cudaMalloc(&shared.q, (size_t)H * D * 2));
            CUDA_CHECK(cudaMemcpy(shared.q, q.data(), (size_t)H * D * 2, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&shared.out, (size_t)H * D * 2));
            CUDA_CHECK(cudaMalloc(&shared.ws, (size_t)H * S * (D + 2) * 4));
            for (int r = 0; r < copies; ++r) {
                bufs[r].q = shared.q; bufs[r].out = shared.out; bufs[r].ws = shared.ws;
            }

            cudaEvent_t e0, e1;
            CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
            for (int w = 0; w < 3; ++w) launch(f, bufs[w % copies], H, L, D, S);
            CUDA_CHECK(cudaDeviceSynchronize());
            double best_ms = 1e30;
            for (int rep = 0; rep < 20; ++rep) {
                const DeviceBufs& b = bufs[rep % copies];
                CUDA_CHECK(cudaEventRecord(e0));
                launch(f, b, H, L, D, S);
                CUDA_CHECK(cudaEventRecord(e1));
                CUDA_CHECK(cudaEventSynchronize(e1));
                float ms = 0.f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                best_ms = std::min(best_ms, (double)ms);
            }
            CUDA_CHECK(cudaGetLastError());
            const double gbs  = (double)copy_bytes / (best_ms * 1e6);
            const double mbu  = gbs / ceil_gbs;
            const double toks = 1e3 / best_ms;
            if (f == Fmt::F16) f16_ms = best_ms; // vs-f16 column stays vs the plain baseline
            std::printf("| %s | %-6d | %7.1fMB | %7.3f | %6.1f | %4.1f%% | %6.0f | %5.2fx |\n",
                        fmt_name(f), L, (double)copy_bytes / 1e6, best_ms, gbs, mbu * 100.0,
                        toks, f16_ms / best_ms);
            CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
            for (int r = 0; r < copies; ++r) {
                CUDA_CHECK(cudaFree(bufs[r].K)); CUDA_CHECK(cudaFree(bufs[r].V));
                if (bufs[r].scK) CUDA_CHECK(cudaFree(bufs[r].scK));
                if (bufs[r].scV) CUDA_CHECK(cudaFree(bufs[r].scV));
            }
            CUDA_CHECK(cudaFree(shared.q)); CUDA_CHECK(cudaFree(shared.out)); CUDA_CHECK(cudaFree(shared.ws));
        }
    }
    std::printf("\nbytes = compressed K+V payload + fp16 group scales; q/out/workspace excluded (<0.1%%).\n");
    std::printf("KV working set rotated across up to 16 copies (>=256MB) so L2 cannot serve the stream.\n");
}

int main(int argc, char** argv) {
    const std::string arg = argc > 1 ? argv[1] : "";
    if (arg == "--bench") {
        int rc = selftest();
        if (rc) return rc;
        std::printf("\n");
        bench();
        return 0;
    }
    if (arg == "--probe") {
        std::printf("measured read ceiling: %.1f GB/s\n", probe_ceiling());
        return 0;
    }
    return selftest();
}
