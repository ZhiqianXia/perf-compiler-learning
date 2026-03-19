#include <cuda_runtime.h>

__global__ void gather_kernel(const float* input, const int* indices, float* output,
                               int n, int axis_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Get index for current position
    int input_idx = (idx / axis_size) * axis_size + indices[idx % axis_size];
    output[idx] = input[input_idx];
}

void gather(const float* input, const int* indices, float* output, int n,
            int axis_size, cudaStream_t stream = 0) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    gather_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        input, indices, output, n, axis_size);
}
