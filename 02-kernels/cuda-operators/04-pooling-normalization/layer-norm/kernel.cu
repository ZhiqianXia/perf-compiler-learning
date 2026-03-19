#include <cuda_runtime.h>
#include <cmath>

__global__ void layer_norm_kernel(const float* x, const float* gamma, const float* beta,
                                   float* y, int batch_size, int hidden_size, float eps) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    extern __shared__ float sdata[];
    float* mean_s = sdata;
    float* var_s = (float*)&sdata[blockDim.x];

    // Compute mean
    float thread_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        thread_sum += x[bid * hidden_size + i];
    }
    mean_s[tid] = thread_sum;
    __syncthreads();

    // Parallel reduction for mean
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            mean_s[tid] += mean_s[tid + s];
        }
        __syncthreads();
    }

    float mean = mean_s[0] / hidden_size;

    // Compute variance
    float thread_var = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float diff = x[bid * hidden_size + i] - mean;
        thread_var += diff * diff;
    }
    var_s[tid] = thread_var;
    __syncthreads();

    // Parallel reduction for variance
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            var_s[tid] += var_s[tid + s];
        }
        __syncthreads();
    }

    float var = var_s[0] / hidden_size;
    float std = sqrtf(var + eps);

    // Normalize and apply gamma, beta
    for (int i = tid; i < hidden_size; i += stride) {
        y[bid * hidden_size + i] =
            ((x[bid * hidden_size + i] - mean) / std) * gamma[i] + beta[i];
    }
}

void layer_norm(const float* x, const float* gamma, const float* beta, float* y,
                int batch_size, int hidden_size, float eps = 1e-5f,
                cudaStream_t stream = 0) {
    int threadsPerBlock = 256;
    size_t shared_mem_size = 2 * threadsPerBlock * sizeof(float);
    layer_norm_kernel<<<batch_size, threadsPerBlock, shared_mem_size, stream>>>(
        x, gamma, beta, y, batch_size, hidden_size, eps);
}
