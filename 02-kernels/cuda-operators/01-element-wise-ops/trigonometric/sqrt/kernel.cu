#include <cuda_runtime.h>
#include <cmath>

__global__ void sqrt_kernel(const float* a, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = sqrtf(a[idx]);
    }
}

void sqrt(const float* a, float* c, int n, cudaStream_t stream = 0) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    sqrt_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a, c, n);
}
