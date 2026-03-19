#include <cuda_runtime.h>
#include <cmath>

__global__ void pow_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = powf(a[idx], b[idx]);
    }
}

void pow(const float* a, const float* b, float* c, int n, cudaStream_t stream = 0) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    pow_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a, b, c, n);
}
