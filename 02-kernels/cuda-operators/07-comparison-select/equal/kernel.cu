#include <cuda_runtime.h>

__global__ void equal_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = (a[idx] == b[idx]) ? 1.0f : 0.0f;
    }
}

void equal(const float* a, const float* b, float* c, int n, cudaStream_t stream = 0) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    equal_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a, b, c, n);
}
