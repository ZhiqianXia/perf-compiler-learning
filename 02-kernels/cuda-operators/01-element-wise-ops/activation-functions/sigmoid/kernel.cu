#include <cuda_runtime.h>
#include <cmath>

__global__ void sigmoid_kernel(const float* a, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Numerically stable sigmoid
        float x = a[idx];
        if (x >= 0.0f) {
            c[idx] = 1.0f / (1.0f + expf(-x));
        } else {
            float ex = expf(x);
            c[idx] = ex / (1.0f + ex);
        }
    }
}

void sigmoid(const float* a, float* c, int n, cudaStream_t stream = 0) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a, c, n);
}
