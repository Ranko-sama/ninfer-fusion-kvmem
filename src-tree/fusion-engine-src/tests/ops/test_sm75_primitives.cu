#include "ops/common/mma.cuh"
#include "ops/common/math.cuh"
#include <cuda_runtime.h>
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
    if (mode <= 2) {
        unsigned a0 = pack16(a + g*16 + 2*t, 1, mode == 1);
        unsigned a1 = pack16(a + (g+8)*16 + 2*t, 1, mode == 1);
        unsigned a2 = pack16(a + g*16 + 2*t+8, 1, mode == 1);
        unsigned a3 = pack16(a + (g+8)*16 + 2*t+8, 1, mode == 1);
        unsigned b0 = pack16(b + (2*t)*8 + g, 8, mode == 1);
        unsigned b1 = pack16(b + (2*t+8)*8 + g, 8, mode == 1);
        if (mode == 0) ninfer::ops::mma_f16(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1);
        if (mode == 1) ninfer::ops::mma_bf16(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1);
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
    for(int mode=0;mode<5;++mode) for(int trial=0;trial<12;++trial) {
        int k=mode==3?32:(mode==4?8:16);
        for(int i=0;i<16*k;++i)a[i]=float((i*17+trial*7)%31-15)/(mode==3?1:8);
        for(int i=0;i<k*8;++i)b[i]=float((i*11+trial*3)%29-14)/(mode==3?1:8);
        if((mode==1||mode==4)&&trial>=6) {
            // Inputs are exactly BF16/TF32 representable but outside FP16's exponent range.
            for(int i=0;i<16*k;++i)a[i]=std::ldexp(a[i],trial%2?40:-40);
            for(int i=0;i<k*8;++i)b[i]=std::ldexp(b[i],trial%2?-40:40);
        }
        matrix<<<1,32>>>(a,b,out,mode);check(cudaGetLastError());check(cudaDeviceSynchronize());
        for(int row=0;row<16;++row)for(int col=0;col<8;++col) {
            double expected=0;for(int j=0;j<k;++j)expected+=double(a[row*k+j])*double(b[j*8+col]);
            double tolerance=mode==2?0.03:0.00001;
            if(!std::isfinite(out[row*8+col])||std::abs(out[row*8+col]-expected)>tolerance) {
                if(failures++<8)std::printf("FAIL mode=%d trial=%d row=%d col=%d actual=%g ref=%g\n",mode,trial,row,col,out[row*8+col],expected);
            }
        }
    }
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
