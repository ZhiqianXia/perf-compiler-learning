#include <cuda_runtime.h>

__global__ void where_kernel(const float* cond, const float* a, const float* b,
                              float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = (cond[idx] != 0.0f) ? a[idx] : b[idx];
    }
}

void where(const float* cond, const float* a, const float* b, float* c, int n,
           cudaStream_t stream = 0) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    where_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(cond, a, b, c, n);
}
