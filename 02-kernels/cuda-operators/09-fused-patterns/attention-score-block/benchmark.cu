#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

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
                           cudaStream_t stream = 0);

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t status = (call);                                              \
        if (status != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error: %s at %s:%d\n",                   \
                         cudaGetErrorString(status), __FILE__, __LINE__);         \
            return 1;                                                             \
        }                                                                         \
    } while (0)

int main() {
    constexpr int batch_size = 8;
    constexpr int num_heads = 8;
    constexpr int seq_len = 128;
    constexpr int head_dim = 64;
    constexpr int warmup = 10;
    constexpr int iterations = 50;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> host_q(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> host_k(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> host_mask(seq_len * seq_len, 0.0f);
    for (float& value : host_q) {
        value = dist(rng);
    }
    for (float& value : host_k) {
        value = dist(rng);
    }
    for (int row = 0; row < seq_len; ++row) {
        for (int col = row + 1; col < seq_len; ++col) {
            host_mask[row * seq_len + col] = -1e9f;
        }
    }

    float* dev_q = nullptr;
    float* dev_k = nullptr;
    float* dev_mask = nullptr;
    float* dev_scores = nullptr;
    float* dev_probs = nullptr;
    CUDA_CHECK(cudaMalloc(&dev_q, host_q.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_k, host_k.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_mask, host_mask.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_scores,
                          batch_size * num_heads * seq_len * seq_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_probs,
                          batch_size * num_heads * seq_len * seq_len * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev_q, host_q.data(), host_q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_k, host_k.data(), host_k.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_mask,
                          host_mask.data(),
                          host_mask.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) {
        attention_score_block(dev_q,
                              dev_k,
                              dev_mask,
                              dev_scores,
                              dev_probs,
                              batch_size,
                              num_heads,
                              seq_len,
                              head_dim,
                              scale);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        attention_score_block(dev_q,
                              dev_k,
                              dev_mask,
                              dev_scores,
                              dev_probs,
                              batch_size,
                              num_heads,
                              seq_len,
                              head_dim,
                              scale);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::printf("attention_score_block b=%d h=%d s=%d d=%d avg_ms=%.4f\n",
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                milliseconds / iterations);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_q);
    cudaFree(dev_k);
    cudaFree(dev_mask);
    cudaFree(dev_scores);
    cudaFree(dev_probs);
    return 0;
}
