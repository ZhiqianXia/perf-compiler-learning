#include <cuda_runtime.h>

#include <cmath>

__global__ void layernorm_residual_kernel(const float* x,
                                          const float* residual,
                                          const float* gamma,
                                          const float* beta,
                                          float* output,
                                          int batch_size,
                                          int hidden_size,
                                          float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= batch_size) {
        return;
    }

    extern __shared__ float shared[];
    float* sum_shared = shared;
    float* sq_sum_shared = shared + blockDim.x;

    float thread_sum = 0.0f;
    float thread_sq_sum = 0.0f;
    for (int col = tid; col < hidden_size; col += blockDim.x) {
        const float value = x[row * hidden_size + col] + residual[row * hidden_size + col];
        thread_sum += value;
        thread_sq_sum += value * value;
    }

    sum_shared[tid] = thread_sum;
    sq_sum_shared[tid] = thread_sq_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_shared[tid] += sum_shared[tid + stride];
            sq_sum_shared[tid] += sq_sum_shared[tid + stride];
        }
        __syncthreads();
    }

    const float mean = sum_shared[0] / hidden_size;
    const float variance = sq_sum_shared[0] / hidden_size - mean * mean;
    const float inv_std = rsqrtf(variance + eps);

    for (int col = tid; col < hidden_size; col += blockDim.x) {
        const int index = row * hidden_size + col;
        const float fused = x[index] + residual[index];
        output[index] = (fused - mean) * inv_std * gamma[col] + beta[col];
    }
}

void layernorm_residual(const float* x,
                        const float* residual,
                        const float* gamma,
                        const float* beta,
                        float* output,
                        int batch_size,
                        int hidden_size,
                        float eps = 1e-5f,
                        cudaStream_t stream = 0) {
    const int threads = 256;
    const size_t shared_mem_size = 2 * threads * sizeof(float);
    layernorm_residual_kernel<<<batch_size, threads, shared_mem_size, stream>>>(
        x, residual, gamma, beta, output, batch_size, hidden_size, eps);
}
