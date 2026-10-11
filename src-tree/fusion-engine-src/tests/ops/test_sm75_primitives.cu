#include "ops/common/mma.cuh"
#include "ops/common/gdn_mma.cuh"
#include "ops/common/math.cuh"
#include "ops/kv_cache/int4_f16_sm75.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
void check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); }
}
__device__ unsigned pack16(const float* p, int stride, bool bf) {
    unsigned lo = bf ? __bfloat16_as_ushort(__float2bfloat16(p[0])) : __half_as_ushort(__float2half(p[0]));
    unsigned hi = bf ? __bfloat16_as_ushort(__float2bfloat16(p[stride])) : __half_as_ushort(__float2half(p[stride]));
    return lo | (hi << 16);
}
__global__ void matrix(const float* a, const float* b, float* out, int mode) {
    const int lane = threadIdx.x, g = lane / 4, t = lane % 4;
    float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    if (mode <= 2 || mode == 5) {
        const bool bf = mode == 1 || mode == 5;
        unsigned a0 = pack16(a + g*16 + 2*t, 1, bf);
        unsigned a1 = pack16(a + (g+8)*16 + 2*t, 1, bf);
        unsigned a2 = pack16(a + g*16 + 2*t+8, 1, bf);
        unsigned a3 = pack16(a + (g+8)*16 + 2*t+8, 1, bf);
        unsigned b0 = pack16(b + (2*t)*8 + g, 8, bf);
        unsigned b1 = pack16(b + (2*t+8)*8 + g, 8, bf);
        if (mode == 0) ninfer::ops::mma_f16(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1);
        if (mode == 1) ninfer::ops::mma_bf16(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1);
        if (mode == 5) ninfer::ops::mma_bf16_gdn(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1);
        if (mode == 2) {
            unsigned d0=0, d1=0;
            ninfer::ops::mma_f16_f16acc(d0,d1,a0,a1,a2,a3,b0,b1);
            c0=__half2float(__ushort_as_half(d0)); c1=__half2float(__ushort_as_half(d0>>16));
            c2=__half2float(__ushort_as_half(d1)); c3=__half2float(__ushort_as_half(d1>>16));
        }
    } else if (mode == 3) {
        unsigned ar[4]={}, br[2]={};
        for(int j=0;j<4;++j) {
            ar[0] |= (unsigned)(unsigned char)(int)a[g*32+4*t+j] << (8*j);
            ar[1] |= (unsigned)(unsigned char)(int)a[(g+8)*32+4*t+j] << (8*j);
            ar[2] |= (unsigned)(unsigned char)(int)a[g*32+4*t+j+16] << (8*j);
            ar[3] |= (unsigned)(unsigned char)(int)a[(g+8)*32+4*t+j+16] << (8*j);
            br[0] |= (unsigned)(unsigned char)(int)b[(4*t+j)*8+g] << (8*j);
            br[1] |= (unsigned)(unsigned char)(int)b[(4*t+j+16)*8+g] << (8*j);
        }
        int i0=0,i1=0,i2=0,i3=0;
        ninfer::ops::mma_s8(i0,i1,i2,i3,ar[0],ar[1],ar[2],ar[3],br[0],br[1]);
        c0=i0;c1=i1;c2=i2;c3=i3;
    } else if (mode == 6) {
        ninfer::ops::mma_tf32_gdn(c0,c1,c2,c3,
            __float_as_uint(a[g*8+t]), __float_as_uint(a[(g+8)*8+t]),
            __float_as_uint(a[g*8+t+4]), __float_as_uint(a[(g+8)*8+t+4]),
            __float_as_uint(b[t*8+g]), __float_as_uint(b[(t+4)*8+g]));
    } else {
        ninfer::ops::mma_tf32(c0,c1,c2,c3,a[g*8+t],a[(g+8)*8+t],a[g*8+t+4],a[(g+8)*8+t+4],b[t*8+g],b[(t+4)*8+g]);
    }
    out[g*8+2*t]=c0;out[g*8+2*t+1]=c1;
    out[(g+8)*8+2*t]=c2;out[(g+8)*8+2*t+1]=c3;
}
template<int Bytes> __global__ void staging(const unsigned char* in, unsigned char* out, int valid) {
    __shared__ __align__(16) unsigned char tile[32*16];
    ninfer::ops::cp_async_zfill<Bytes>(tile + threadIdx.x*16, in + threadIdx.x*16, valid);
    ninfer::ops::cp_commit();ninfer::ops::cp_wait<0>();__syncthreads();
    const int other=(threadIdx.x+1)%32;
    for(int i=0;i<Bytes;++i)out[threadIdx.x*16+i]=tile[other*16+i];
}
__global__ void convert_bf16(const unsigned* input, unsigned* output) {
    const int i = threadIdx.x;
    output[i] = ninfer::ops::pack_bf16x2(__uint_as_float(input[2*i]),
                                       __uint_as_float(input[2*i+1]));
}
float reference_tf32(float input) {
    unsigned bits;
    std::memcpy(&bits, &input, sizeof(bits));
    unsigned retained = bits >> 13, discarded = bits & 8191U;
    if (discarded > 4096U || (discarded == 4096U && (retained & 1U))) ++retained;
    bits = retained << 13;
    float output;
    std::memcpy(&output, &bits, sizeof(output));
    return output;
}
__global__ void int4_half_scale(unsigned short* output) {
    const unsigned scale = blockIdx.x*blockDim.x+threadIdx.x;
    if (scale >= 65536) return;
    for (unsigned group=0;group<2;++group) {
        const unsigned codes = group == 0 ? 0x76543210u : 0xfedcba98u;
        const int4 actual = ninfer::ops::kv_cache_int4_dequant_f16x8_half_scale(
            reinterpret_cast<const std::uint8_t*>(&codes), __ushort_as_half(scale));
        const auto* halves = reinterpret_cast<const unsigned short*>(&actual);
        for (int i=0;i<8;++i) output[scale*16+group*8+i] = halves[i];
    }
}
unsigned reference_bf16(unsigned bits) {
    if ((bits & 0x7fffffffU) > 0x7f800000U) return 0x7fffU;
    // Independent integer ties-to-even rounding at the public BF16 boundary.
    unsigned retained = bits >> 16, discarded = bits & 65535U;
    if (discarded > 32768U || (discarded == 32768U && (retained & 1U))) ++retained;
    return retained;
}
}

int main() {
    float *a,*b,*out;
    check(cudaMallocManaged(&a,512*sizeof(float)));check(cudaMallocManaged(&b,256*sizeof(float)));
    check(cudaMallocManaged(&out,128*sizeof(float)));
    int failures=0;
    unsigned *cast_input, *cast_output;
    check(cudaMallocManaged(&cast_input,64*sizeof(unsigned)));
    check(cudaMallocManaged(&cast_output,32*sizeof(unsigned)));
    const unsigned edges[] = {0U,0x80000000U,1U,0x80000001U,0x00008000U,0x00018000U,
        0x3f808000U,0x3f818000U,0xbf808001U,0x7f7fffffU,0x7f800000U,0xff800000U,
        0x7fc00000U,0xff800001U,0x00800000U,0x007fffffU};
    for(int i=0;i<64;++i) cast_input[i]=i<16?edges[i]:0x9e3779b9U*unsigned(i);
    convert_bf16<<<1,32>>>(cast_input,cast_output);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    for(int i=0;i<32;++i) {
        const unsigned expected=reference_bf16(cast_input[2*i])|(reference_bf16(cast_input[2*i+1])<<16);
        if(cast_output[i]!=expected) {
            ++failures;std::printf("BF16 cast FAIL pair=%d actual=%08x expected=%08x\n",i,cast_output[i],expected);
        }
    }
    cudaFree(cast_input);cudaFree(cast_output);
    for(int mode=0;mode<7;++mode) for(int trial=0;trial<(mode==5?40:(mode==6?32:12));++trial) {
        int k=mode==3?32:((mode==4||mode==6)?8:16);
        for(int i=0;i<16*k;++i)a[i]=float((i*17+trial*7)%31-15)/(mode==3?1:8);
        for(int i=0;i<k*8;++i)b[i]=float((i*11+trial*3)%29-14)/(mode==3?1:8);
        if((mode==1||mode==4||mode==5)&&trial>=6) {
            // Inputs are exactly BF16/TF32 representable but outside FP16's exponent range.
            for(int i=0;i<16*k;++i)a[i]=std::ldexp(a[i],trial%2?40:-40);
            for(int i=0;i<k*8;++i)b[i]=std::ldexp(b[i],trial%2?-40:40);
        }
        if(mode==5&&trial>=12) {
            // Tiny BF16 operands exercise the bounded native path; wide exponents
            // and values too large for it retain the original full-range fallback.
            for(int i=0;i<16*k;++i) a[i]=std::ldexp(float((i*17+trial*7)%31-15)/8,
                trial<18?-20:((i%3==0)?40:-40));
            for(int i=0;i<k*8;++i) b[i]=std::ldexp(float((i*11+trial*3)%29-14)/8,
                trial<18?-12:((i%3==0)?-40:40));
        }
        if(mode==5&&trial>=24) {
            // Full BF16 significands near the tiny-value boundary
            // and at the largest accepted exponent, including mixed signs.
            for(int i=0;i<16*k;++i) a[i]=std::ldexp(
                float(128+(i*73+trial*17)%128)/128*(i%2?-1:1),
                trial%2 ? 5 : -27+(i%4));
            for(int i=0;i<k*8;++i) b[i]=std::ldexp(
                float(128+(i*31+trial*11)%128)/128*(i%3?-1:1),
                trial%2 ? -27+(i%4) : 5);
        }
        if(mode==6) {
            // Full FP32 mantissas, cancellation, uniform and mixed exponents.
            for(int i=0;i<16*k;++i) a[i]=std::ldexp(float((i*11939+trial*3137)%1048573-524286)/524286.0f,
                trial<8?0:(trial<16?-12:((i%3==0)?40:-40)));
            for(int i=0;i<k*8;++i) b[i]=std::ldexp(float((i*31627+trial*1327)%1048573-524286)/524286.0f,
                trial<8?0:(trial<16?-12:((i%3==0)?-40:40)));
        }
        if((mode==5&&trial>=32)||(mode==6&&trial>=24)) {
            // Mixed exponent bands inside a warp, including whole tiny rows
            // and columns: qualify their absolute error against the cast bound.
            for(int i=0;i<16*k;++i) a[i]=std::ldexp(
                float(128+(i*73+trial*17)%128)/128*(i%2?-1:1),
                (i/k)%3==0?-45:0);
            for(int i=0;i<k*8;++i) b[i]=std::ldexp(
                float(128+(i*31+trial*11)%128)/128*(i%3?-1:1),
                i%8<3?-40:0);
        }
        matrix<<<1,32>>>(a,b,out,mode);check(cudaGetLastError());check(cudaDeviceSynchronize());
        bool bounded = mode>=5;
        for(int i=0;i<16*k;++i) bounded = bounded && std::abs(a[i])<=65280;
        for(int i=0;i<k*8;++i) bounded = bounded && std::abs(b[i])<=65280;
        for(int row=0;row<16;++row)for(int col=0;col<8;++col) {
            double expected=0, magnitude=0, cast_bound=0;
            for(int j=0;j<k;++j) {
                const float av = mode==6 ? reference_tf32(a[row*k+j]) : a[row*k+j];
                const float bv = mode==6 ? reference_tf32(b[j*8+col]) : b[j*8+col];
                const double term=double(av)*double(bv);
                expected+=term;magnitude+=std::abs(term);
                // Analytic FP16 subnormal rounding bound for represented BF16/TF32 inputs.
                // This private primitive bound does not change public Op criteria.
                if(bounded) {
                    const double epsilon=mode==6 ? 0x1p-24 : 0x1p-25;
                    cast_bound += epsilon*(std::abs(double(av))+std::abs(double(bv)))+epsilon*epsilon;
                }
            }
            double tolerance=mode==2?0.03:0.00001;
            if(mode>=5) tolerance=std::max(1e-35, magnitude*1e-6)+cast_bound;
            if(!std::isfinite(out[row*8+col])||std::abs(out[row*8+col]-expected)>tolerance) {
                if(failures++<8)std::printf("FAIL mode=%d trial=%d row=%d col=%d actual=%g ref=%g\n",mode,trial,row,col,out[row*8+col],expected);
            }
        }
    }
    unsigned short* dequant;
    check(cudaMallocManaged(&dequant,65536*16*sizeof(unsigned short)));
    int4_half_scale<<<256,256>>>(dequant);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    for(unsigned scale=0;scale<65536;++scale) {
        if((scale&0x7c00u)==0x7c00u) continue;
        const double represented_scale=__half2float(__ushort_as_half(scale));
        for(int code=0;code<16;++code) {
            const double expected=double(code<8?code:code-16)*represented_scale;
            const unsigned short bits=__half_as_ushort(__float2half_rn(float(expected)));
            if(dequant[scale*16+code]!=bits) {
                if(failures++<8)std::printf("int4 FP16 scale FAIL scale=%u code=%d actual=%04x ref=%04x\n",
                    scale,code,dequant[scale*16+code],bits);
            }
        }
    }
    cudaFree(dequant);
    unsigned char *in,*copied;check(cudaMallocManaged(&in,512));check(cudaMallocManaged(&copied,512));
    for(int i=0;i<512;++i)in[i]=(unsigned char)(i*13+7);
    for(int bytes: {4,8,16})for(int valid=0;valid<=bytes;++valid) {
        if(bytes==4)staging<4><<<1,32>>>(in,copied,valid);
        if(bytes==8)staging<8><<<1,32>>>(in,copied,valid);
        if(bytes==16)staging<16><<<1,32>>>(in,copied,valid);
        check(cudaGetLastError());check(cudaDeviceSynchronize());
        for(int lane=0;lane<32;++lane)for(int i=0;i<bytes;++i)
            failures+=copied[lane*16+i]!=(i<valid?in[((lane+1)%32)*16+i]:0);
    }
    cudaFree(a);cudaFree(b);cudaFree(out);cudaFree(in);cudaFree(copied);
    std::printf("sm75 primitive oracle: %s (%d mismatches)\n",failures?"FAIL":"PASS",failures);
    return failures?1:0;
}
