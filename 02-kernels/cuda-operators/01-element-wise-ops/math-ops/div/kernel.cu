#include <cuda_runtime.h>

__global__ void div_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = (b[idx] != 0.0f) ? a[idx] / b[idx] : 0.0f;
    }
}

void div(const float* a, const float* b, float* c, int n, cudaStream_t stream = 0) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    div_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a, b, c, n);
}
