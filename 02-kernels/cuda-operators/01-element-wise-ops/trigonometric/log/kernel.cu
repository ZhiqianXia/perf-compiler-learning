#include <cuda_runtime.h>
#include <cmath>

__global__ void log_kernel(const float* a, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] > 0.0f ? logf(a[idx]) : -INFINITY;
    }
}

void log(const float* a, float* c, int n, cudaStream_t stream = 0) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    log_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a, c, n);
}
