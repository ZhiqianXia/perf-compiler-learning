#include <cuda_runtime.h>

#include <cmath>

namespace {

__global__ void qk_scores_kernel(const float* q,
                                 const float* k,
                                 float* scores,
                                 int batch_size,
                                 int num_heads,
                                 int seq_len,
                                 int head_dim) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int linear_index = blockIdx.z;

    const int batch_index = linear_index / num_heads;
    const int head_index = linear_index % num_heads;
    if (batch_index >= batch_size || row >= seq_len || col >= seq_len) {
        return;
    }

    float sum = 0.0f;
    const int q_base = ((batch_index * num_heads + head_index) * seq_len + row) * head_dim;
    const int k_base = ((batch_index * num_heads + head_index) * seq_len + col) * head_dim;
    for (int d = 0; d < head_dim; ++d) {
        sum += q[q_base + d] * k[k_base + d];
    }

    const int score_index =
        ((batch_index * num_heads + head_index) * seq_len + row) * seq_len + col;
    scores[score_index] = sum;
}

__global__ void scale_mask_softmax_kernel(const float* scores,
                                          const float* mask,
                                          float* probs,
                                          int batch_size,
                                          int num_heads,
                                          int seq_len,
                                          float scale) {
    const int row = blockIdx.x;
    const int linear_index = blockIdx.y;
    const int tid = threadIdx.x;

    const int batch_index = linear_index / num_heads;
    const int head_index = linear_index % num_heads;
    if (batch_index >= batch_size || row >= seq_len) {
        return;
    }

    const int row_offset = ((batch_index * num_heads + head_index) * seq_len + row) * seq_len;

    extern __shared__ float shared[];
    float* max_shared = shared;
    float* sum_shared = shared + blockDim.x;

    float thread_max = -CUDART_INF_F;
    for (int col = tid; col < seq_len; col += blockDim.x) {
        float value = scores[row_offset + col] * scale;
        if (mask != nullptr) {
            value += mask[row * seq_len + col];
        }
        thread_max = fmaxf(thread_max, value);
    }
    max_shared[tid] = thread_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            max_shared[tid] = fmaxf(max_shared[tid], max_shared[tid + stride]);
        }
        __syncthreads();
    }

    const float row_max = max_shared[0];
    float thread_sum = 0.0f;
    for (int col = tid; col < seq_len; col += blockDim.x) {
        float value = scores[row_offset + col] * scale;
        if (mask != nullptr) {
            value += mask[row * seq_len + col];
        }
        thread_sum += expf(value - row_max);
    }
    sum_shared[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_shared[tid] += sum_shared[tid + stride];
        }
        __syncthreads();
    }

    const float row_sum = sum_shared[0];
    for (int col = tid; col < seq_len; col += blockDim.x) {
        float value = scores[row_offset + col] * scale;
        if (mask != nullptr) {
            value += mask[row * seq_len + col];
        }
        probs[row_offset + col] = expf(value - row_max) / row_sum;
    }
}

}  // namespace

void attention_score_block(const float* q,
                           const float* k,
                           const float* mask,
                           float* scores,
                           float* probs,
                           int batch_size,
                           int num_heads,
                           int seq_len,
                           int head_dim,
                           float scale,
                           cudaStream_t stream = 0) {
    dim3 block_dim_scores(16, 16);
    dim3 grid_dim_scores((seq_len + block_dim_scores.x - 1) / block_dim_scores.x,
                         (seq_len + block_dim_scores.y - 1) / block_dim_scores.y,
                         batch_size * num_heads);
    qk_scores_kernel<<<grid_dim_scores, block_dim_scores, 0, stream>>>(
        q, k, scores, batch_size, num_heads, seq_len, head_dim);

    const int threads = 256;
    dim3 grid_dim_softmax(seq_len, batch_size * num_heads);
    const size_t shared_mem_size = 2 * threads * sizeof(float);
    scale_mask_softmax_kernel<<<grid_dim_softmax, threads, shared_mem_size, stream>>>(
        scores, mask, probs, batch_size, num_heads, seq_len, scale);
}
