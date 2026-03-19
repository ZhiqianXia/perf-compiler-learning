#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* a, float* c, int n, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] > 0.0f ? a[idx] : (alpha * a[idx]);
    }
}

void leaky_relu(const float* a, float* c, int n, float alpha = 0.01f, cudaStream_t stream = 0) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a, c, n, alpha);
}
