#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::ops {

// Signed int4 codes and a represented FP16 scale. Both the integer decode and
// the subtraction are exact; the product has one FP16 rounding, just like an
// exact FP32 code*scale followed by the cache consumer's FP16 cast.
__device__ __forceinline__ int4 kv_cache_int4_dequant_f16x8_half_scale(
    const std::uint8_t* packed4, __half scale) {
    const unsigned raw = *reinterpret_cast<const unsigned*>(packed4);
    const half2 offset = __float2half2_rn(1032.0f);
    const half2 scales = __half2half2(scale);
    unsigned out[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned bits = 0x64006400u |
            ((((raw >> (8*i)) & 15u) ^ 8u)) |
            ((((raw >> (8*i+4)) & 15u) ^ 8u) << 16);
        const half2 biased = *reinterpret_cast<const half2*>(&bits);
        const half2 value = __hmul2(__hsub2(biased, offset), scales);
        out[i] = *reinterpret_cast<const unsigned*>(&value);
    }
    return make_int4(out[0], out[1], out[2], out[3]);
}

} // namespace ninfer::ops
