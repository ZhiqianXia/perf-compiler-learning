#include <cuda_runtime.h>
#include <cmath>

__global__ void gelu_kernel(const float* a, float* c, int n) {
    float sqrt2pi = 1.12837916709551f;  // sqrt(2/π)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = a[idx];
        float cdf = 0.5f * (1.0f + tanhf(sqrt2pi * (x + 0.044715f * x * x * x)));
        c[idx] = x * cdf;
    }
}

void gelu(const float* a, float* c, int n, cudaStream_t stream = 0) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    gelu_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a, c, n);
}
