#include <cuda_runtime.h>

__global__ void concat_kernel(const float* const* inputs, const int* strides,
                               float* output, int total_elements, int num_inputs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Determine which input and offset
    int remaining = idx;
    for (int i = 0; i < num_inputs; i++) {
        if (remaining < strides[i + 1] - strides[i]) {
            output[idx] = inputs[i][remaining];
            return;
        }
        remaining -= (strides[i + 1] - strides[i]);
    }
}

void concat(const float* const* inputs, float* output, const int* strides,
            int num_inputs, int total_elements, cudaStream_t stream = 0) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    concat_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        inputs, strides, output, total_elements, num_inputs);
}
