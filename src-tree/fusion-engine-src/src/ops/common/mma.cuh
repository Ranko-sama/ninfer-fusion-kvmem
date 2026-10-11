#pragma once

#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ninfer::ops {

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2,
                                              unsigned& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

#if defined(NINFER_SM75)
// --- Turing (sm_75) tensor-core stand-ins --------------------------------------------------------
//
// Turing has f16 and int8 tensor cores but only the m16n8k8 (f16) and m8n8k16 (s8) shapes; the
// m16n8k16/m16n8k32 shapes every kernel in this tree uses arrived with Ampere (sm_80). Each
// emulation below splits its operands along the shape the hardware does have, using the register
// fragment layouts PTX specifies, which compose exactly:
//
//   * m16n8k16 (f16/bf16) = two m16n8k8 over the k halves. The A fragment is
//     {a0=(r0..7,k0..7), a1=(r8..15,k0..7), a2=(r0..7,k8..15), a3=(r8..15,k8..15)}, and an
//     m16n8k8 takes (rows0..7, rows8..15) as its A, so (a0,a1) and (a2,a3) are the two steps with
//     B halves b0 and b1. BF16 uses this route only when every operand survives the FP16 cast
//     exactly; otherwise a warp-uniform FP32 fallback preserves the represented BF16 values.
//
//   * m16n8k32 (s8) = four m8n8k16. The A fragment holds {a0=(r0..7,k0..15), a1=(r8..15,k0..15),
//     a2=(r0..7,k16..31), a3=(r8..15,k16..31)} and B {b0=(k0..15), b1=(k16..31)}, and the 8-row
//     halves of the accumulator are (c0,c1) for rows0..7 and (c2,c3) for rows8..15: each m8n8k16
//     takes one row half, one k half and its C pair.
//
// ldmatrix (m8n8.x1/x2/x4) exists on Turing, so fragment assembly is unchanged.
__device__ __forceinline__ unsigned bf162_to_f162(unsigned b) {
    const auto* bf = reinterpret_cast<const __nv_bfloat162*>(&b);
    const half2 h  = __float22half2_rn(__bfloat1622float2(*bf));
    return *reinterpret_cast<const unsigned*>(&h);
}

__device__ __forceinline__ bool bf162_fits_f162(unsigned b) {
    const float lo = __uint_as_float(b << 16);
    const float hi = __uint_as_float(b & 0xffff0000u);
    return __half2float(__float2half_rn(lo)) == lo &&
           __half2float(__float2half_rn(hi)) == hi;
}

// Turing-native f16 MMA, 16x8x8 (two k-steps of the m16n8k16 the callers ask for).
__device__ __forceinline__ void mma_f16_m16n8k8(float& c0, float& c1, float& c2, float& c3,
                                                unsigned a0, unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(b0));
}

// Turing-native f16 MMA with f16 accumulation, 16x8x8.
__device__ __forceinline__ void mma_f16_f16acc_m16n8k8(unsigned& d0, unsigned& d1, unsigned a0,
                                                       unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
                 : "+r"(d0), "+r"(d1)
                 : "r"(a0), "r"(a1), "r"(b0));
}

// Turing-native s8 MMA, 8x8x16.
__device__ __forceinline__ void mma_s8_m8n8k16(int& c0, int& c1, unsigned a0, unsigned b0) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a0), "r"(b0));
}
#endif // NINFER_SM75

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
#if defined(NINFER_SM75)
    const bool exact = bf162_fits_f162(a0) && bf162_fits_f162(a1) && bf162_fits_f162(a2) &&
                       bf162_fits_f162(a3) && bf162_fits_f162(b0) && bf162_fits_f162(b1);
    if (__all_sync(0xffffffffu, exact)) {
        mma_f16_m16n8k8(c0, c1, c2, c3, bf162_to_f162(a0), bf162_to_f162(a1), bf162_to_f162(b0));
        mma_f16_m16n8k8(c0, c1, c2, c3, bf162_to_f162(a2), bf162_to_f162(a3), bf162_to_f162(b1));
    } else {
        const int lane = threadIdx.x & 31;
        const int row = lane >> 2;
        const int col = 2 * (lane & 3);
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            const int src = (k & 7) / 2;
            const unsigned top_bits = __shfl_sync(0xffffffffu, k < 8 ? a0 : a2, row * 4 + src);
            const unsigned bottom_bits = __shfl_sync(0xffffffffu, k < 8 ? a1 : a3, row * 4 + src);
            const unsigned left_bits = __shfl_sync(0xffffffffu, k < 8 ? b0 : b1, col * 4 + src);
            const unsigned right_bits = __shfl_sync(0xffffffffu, k < 8 ? b0 : b1, (col + 1) * 4 + src);
            const int shift = (k & 1) * 16;
            const float top = __uint_as_float(((top_bits >> shift) & 0xffffu) << 16);
            const float bottom = __uint_as_float(((bottom_bits >> shift) & 0xffffu) << 16);
            const float left = __uint_as_float(((left_bits >> shift) & 0xffffu) << 16);
            const float right = __uint_as_float(((right_bits >> shift) & 0xffffu) << 16);
            c0 = fmaf(top, left, c0);
            c1 = fmaf(top, right, c1);
            c2 = fmaf(bottom, left, c2);
            c3 = fmaf(bottom, right, c3);
        }
    }
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
#if defined(NINFER_SM75)
    mma_f16_m16n8k8(c0, c1, c2, c3, a0, a1, b0);
    mma_f16_m16n8k8(c0, c1, c2, c3, a2, a3, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

// FP16 accumulation: twice the FP32-accumulator rate on GeForce parts. d0 holds the row-gid pair
// and d1 the row-gid+8 pair of the fragment, in the same order as mma_f16's c0..c3.
__device__ __forceinline__ void mma_f16_f16acc(unsigned& d0, unsigned& d1, unsigned a0,
                                               unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                               unsigned b1) {
#if defined(NINFER_SM75)
    mma_f16_f16acc_m16n8k8(d0, d1, a0, a1, b0);
    mma_f16_f16acc_m16n8k8(d0, d1, a2, a3, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(d0), "+r"(d1)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if defined(NINFER_SM75)
    // a0=(r0..7,k0..15), a1=(r8..15,k0..15), a2=(r0..7,k16..31), a3=(r8..15,k16..31);
    // b0 covers k0..15 and b1 k16..31; the accumulator halves are (c0,c1) rows0..7 and (c2,c3)
    // rows8..15. Each m8n8k16 therefore takes one row half, one k half and its C pair.
    mma_s8_m8n8k16(c0, c1, a0, b0);
    mma_s8_m8n8k16(c2, c3, a1, b0);
    mma_s8_m8n8k16(c0, c1, a2, b1);
    mma_s8_m8n8k16(c2, c3, a3, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
#if defined(NINFER_SM75)
    const int lane = threadIdx.x & 31;
    const int row = lane >> 2;
    const int col = 2 * (lane & 3);
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float top = __uint_as_float(__shfl_sync(0xffffffffu, k < 4 ? a0 : a2, row * 4 + (k & 3)));
        const float bottom = __uint_as_float(__shfl_sync(0xffffffffu, k < 4 ? a1 : a3, row * 4 + (k & 3)));
        const float left = __uint_as_float(__shfl_sync(0xffffffffu, k < 4 ? b0 : b1, col * 4 + (k & 3)));
        const float right = __uint_as_float(__shfl_sync(0xffffffffu, k < 4 ? b0 : b1, (col + 1) * 4 + (k & 3)));
        c0 = fmaf(top, left, c0);
        c1 = fmaf(top, right, c1);
        c2 = fmaf(bottom, left, c2);
        c3 = fmaf(bottom, right, c3);
    }
#else

    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_tf32(float& c0, float& c1, float& c2, float& c3, float a0,
                                         float a1, float a2, float a3, float b0, float b1) {
    mma_tf32_bits(c0, c1, c2, c3, __float_as_uint(a0), __float_as_uint(a1), __float_as_uint(a2),
                  __float_as_uint(a3), __float_as_uint(b0), __float_as_uint(b1));
}

__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1, unsigned sfa,
                                               unsigned sfb) {
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, "
                 "{%8,%9}, "
                 "{%0,%1,%2,%3}, "
                 "{%10}, "
                 "{%11,%12}, "
                 "{%13}, "
                 "{%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(sfb), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
}

} // namespace ninfer::ops
