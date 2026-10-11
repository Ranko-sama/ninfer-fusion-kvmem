#pragma once

#include "ops/common/mma.cuh"

namespace ninfer::ops {

#if defined(NINFER_SM75)
// Keep the uncommon full-range emulators out of each unrolled MMA site.
// Otherwise their scalar shuffle loops inflate the hot instruction footprint.
static __device__ __noinline__ void gdn_bf16_full_range(
    float& c0, float& c1, float& c2, float& c3, unsigned a0, unsigned a1,
    unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
}
#endif

// Private GDN BF16 products on Turing. Represented normal values within
// FP16 range convert exactly; only the tiny subnormal tail rounds, with
// absolute operand error <= 2^-25. A K-term dot adds at most
// sum(e*(|a|+|b|)+e^2), e=2^-25, before FP32 accumulation error.
// Public GDN output/state criteria against FP64 are unchanged.
// Larger operands and non-finite values retain the full-exponent fallback;
// in particular a large-times-tiny product must never narrow to Inf*0.
__device__ __forceinline__ void mma_bf16_gdn(
    float& c0, float& c1, float& c2, float& c3, unsigned a0, unsigned a1,
    unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if defined(NINFER_SM75)
    const unsigned inputs[6] = {a0, a1, a2, a3, b0, b1};
    bool bounded = true;
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        const unsigned lo = inputs[i] & 0x7fffu;
        const unsigned hi = (inputs[i] >> 16) & 0x7fffu;
        bounded = bounded && lo <= 0x477fu && hi <= 0x477fu;
    }
    if (__all_sync(0xffffffffu, bounded)) {
        unsigned halves[6];
#pragma unroll
        for (int i = 0; i < 6; ++i) {
            const half2 pair = __floats2half2_rn(
                __uint_as_float(inputs[i] << 16),
                __uint_as_float(inputs[i] & 0xffff0000u));
            halves[i] = *reinterpret_cast<const unsigned*>(&pair);
        }
        mma_f16_m16n8k8(c0, c1, c2, c3, halves[0], halves[1], halves[4]);
        mma_f16_m16n8k8(c0, c1, c2, c3, halves[2], halves[3], halves[5]);
    } else {
        gdn_bf16_full_range(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    }
#else
    mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#endif
}

// The two-stage GDN contract uses TF32 residual products. The full-range
// fallback rounds explicitly at that private boundary. The Turing fast route
// combines normal TF32 rounding and FP16 conversion; its subnormal error bound
// relative to represented TF32 is given below.
__device__ __forceinline__ unsigned round_tf32_bits(unsigned bits) {
    if ((bits & 0x7f800000u) == 0x7f800000u) return bits;
    return (bits + 0xfffu + ((bits >> 13) & 1u)) & 0xffffe000u;
}

#if defined(NINFER_SM75)
static __device__ __noinline__ void gdn_tf32_full_range(
    float& c0, float& c1, float& c2, float& c3, unsigned a0, unsigned a1,
    unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    mma_tf32_bits(c0, c1, c2, c3, round_tf32_bits(a0), round_tf32_bits(a1),
                  round_tf32_bits(a2), round_tf32_bits(a3), round_tf32_bits(b0), round_tf32_bits(b1));
}
#endif

__device__ __forceinline__ void mma_tf32_gdn(
    float& c0, float& c1, float& c2, float& c3, unsigned a0, unsigned a1,
    unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if defined(NINFER_SM75)
    // FP16 and TF32 have the same 10 fraction bits in the normal FP16 range:
    // one FP32 -> FP16 RNE conversion performs the required rounding directly.
    // In the subnormal tail, error relative to rounded TF32 is <= 2^-24
    // (FP16 half-ulp 2^-25 plus TF32 half-ulp <= 2^-35).
    const unsigned inputs[6] = {a0, a1, a2, a3, b0, b1};
    bool bounded = true;
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        const unsigned magnitude = inputs[i] & 0x7fffffffu;
        bounded = bounded && magnitude <= 0x477fe000u;
    }
    if (__all_sync(0xffffffffu, bounded)) {
        const unsigned lane = threadIdx.x & 31u;
        const unsigned t = lane & 3u;
        const unsigned lo = (lane & ~3u) | ((2u * t) & 3u);
        const unsigned hi = lo + 1u;
        const unsigned lower[3] = {a0, a1, b0};
        const unsigned upper[3] = {a2, a3, b1};
        unsigned h[3];
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const unsigned x0 = __shfl_sync(0xffffffffu, lower[i], lo);
            const unsigned x1 = __shfl_sync(0xffffffffu, upper[i], lo);
            const unsigned y0 = __shfl_sync(0xffffffffu, lower[i], hi);
            const unsigned y1 = __shfl_sync(0xffffffffu, upper[i], hi);
            const half2 pair = __floats2half2_rn(
                __uint_as_float(t < 2 ? x0 : x1),
                __uint_as_float(t < 2 ? y0 : y1));
            h[i] = *reinterpret_cast<const unsigned*>(&pair);
        }
        mma_f16_m16n8k8(c0, c1, c2, c3, h[0], h[1], h[2]);
    } else {
        gdn_tf32_full_range(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    }
#else
    mma_tf32_bits(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#endif
}

} // namespace ninfer::ops
