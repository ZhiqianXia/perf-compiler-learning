#include <cuda_runtime.h>

__global__ void mul_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

void mul(const float* a, const float* b, float* c, int n, cudaStream_t stream = 0) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    mul_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a, b, c, n);
}
