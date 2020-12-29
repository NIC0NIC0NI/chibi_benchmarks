#include <stdint.h>
#include <algorithm>
#include <cuda.h>

__device__ __forceinline__
int32_t clamp_to_0_a(int32_t x) { 
	return ((-x) >> 31) & x; 
}
__device__ __forceinline__
int32_t clamp_to_255_a(int32_t x) {
	return (((255 - x) >> 31) | x) & 255;
}

__device__ __forceinline__
int32_t clamp_to_0_b(int32_t x) { 
	return (x >= 0) ? x : 0;
}

__device__ __forceinline__
int32_t clamp_to_255_b(int32_t x) {
	return (x <= 255) ? x : 255;
}

#define GRIDDIM  512
#define BLOCKDIM 256

__global__ void clamp_arithmetic_logic_kernel(int4* x4, size_t n4, size_t repetition) {
    const int tid = threadIdx.x + blockIdx.x * BLOCKDIM;
    for(size_t k = 0; k < repetition; ++k) {            // 为了测试GPU的吞吐率，在线程内部进行重复
        for(size_t i = tid; i < n4; i += GRIDDIM * BLOCKDIM) {
            int4 xx = x4[i];
            xx.x = clamp_to_0_a(clamp_to_255_a(xx.x));
            xx.y = clamp_to_0_a(clamp_to_255_a(xx.y));
            xx.z = clamp_to_0_a(clamp_to_255_a(xx.z));
            xx.w = clamp_to_0_a(clamp_to_255_a(xx.w));
            x4[i] = xx;
        }
    }
}

__global__ void clamp_compare_select_kernel(int4* x4, size_t n4, size_t repetition) {
    const int tid = threadIdx.x + blockIdx.x * BLOCKDIM;
    for(size_t k = 0; k < repetition; ++k) {
        for(size_t i = tid; i < n4; i += GRIDDIM * BLOCKDIM) {
            int4 xx = x4[i];
            xx.x = clamp_to_0_b(clamp_to_255_b(xx.x));
            xx.y = clamp_to_0_b(clamp_to_255_b(xx.y));
            xx.z = clamp_to_0_b(clamp_to_255_b(xx.z));
            xx.w = clamp_to_0_b(clamp_to_255_b(xx.w));
            x4[i] = xx;
        }
    }
}

void clamp_arithmetic_logic(int32_t* x, size_t n, size_t repetition) {
    int4* x4 = reinterpret_cast<int4*>(x);
    int n4 = n / 4;
    int grid_dim = std::min(GRIDDIM, (n4 - 1) / BLOCKDIM + 1);
    clamp_arithmetic_logic_kernel<<<grid_dim, BLOCKDIM>>>(x4, n4, repetition);
    cudaDeviceSynchronize();
}

void clamp_compare_select(int32_t* x, size_t n, size_t repetition) {
    int4* x4 = reinterpret_cast<int4*>(x);
    int n4 = n / 4;
    int grid_dim = std::min(GRIDDIM, (n4 - 1) / BLOCKDIM + 1);
    clamp_compare_select_kernel<<<grid_dim, BLOCKDIM>>>(x4, n4, repetition);
    cudaDeviceSynchronize();
}
