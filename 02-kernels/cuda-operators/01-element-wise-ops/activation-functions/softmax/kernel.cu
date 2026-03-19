#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

// Note: Simplified version for single row softmax
__global__ void softmax_kernel(const float* x, float* y, int n) {
    int idx = threadIdx.x;
    int stride = blockDim.x;

    // Shared memory for max and sum
    extern __shared__ float sdata[];
    float* max_val = sdata;
    float* sum_val = (float*)&max_val[1];

    // Find max (numerically stable)
    float thread_max = -INFINITY;
    for (int i = idx; i < n; i += stride) {
        thread_max = fmaxf(thread_max, x[i]);
    }

    max_val[idx] = thread_max;
    __syncthreads();

    // Reduce to find global max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (idx < s) {
            max_val[idx] = fmaxf(max_val[idx], max_val[idx + s]);
        }
        __syncthreads();
    }

    float max_exp = max_val[0];

    // Compute sum of exp
    float thread_sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        thread_sum += expf(x[i] - max_exp);
    }

    sum_val[idx] = thread_sum;
    __syncthreads();

    // Reduce to find sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (idx < s) {
            sum_val[idx] += sum_val[idx + s];
        }
        __syncthreads();
    }

    float sum_exp = sum_val[0];

    // Compute softmax
    for (int i = idx; i < n; i += stride) {
        y[i] = expf(x[i] - max_exp) / sum_exp;
    }
}

void softmax(const float* x, float* y, int n, cudaStream_t stream = 0) {
    int threadsPerBlock = 256;
    size_t shared_mem_size = 2 * threadsPerBlock * sizeof(float);
    softmax_kernel<<<1, threadsPerBlock, shared_mem_size, stream>>>(x, y, n);
}
