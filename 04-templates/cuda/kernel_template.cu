#include <cuda_runtime.h>

__global__ void example_kernel(const float* input, float* output, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        output[index] = input[index];
    }
}
