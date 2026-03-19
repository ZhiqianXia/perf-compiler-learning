#include <cuda_runtime.h>

__global__ void reduce_sum_kernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

void reduce_sum(const float* input, float* output, int n, cudaStream_t stream = 0) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    size_t shared_mem_size = threadsPerBlock * sizeof(float);
    reduce_sum_kernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size, stream>>>(
        input, output, n);
}
