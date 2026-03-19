#include <cuda_runtime.h>

__global__ void relu_kernel(const float* a, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] > 0.0f ? a[idx] : 0.0f;
    }
}

void relu(const float* a, float* c, int n, cudaStream_t stream = 0) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    relu_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a, c, n);
}
