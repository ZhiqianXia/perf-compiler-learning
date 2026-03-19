#include <cuda_runtime.h>

#include <cstdio>
#include <random>
#include <vector>

void matmul_bias_gelu(const float* a,
                      const float* b,
                      const float* bias,
                      float* workspace,
                      float* output,
                      int m,
                      int n,
                      int k,
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
    constexpr int m = 512;
    constexpr int n = 2048;
    constexpr int k = 768;
    constexpr int warmup = 10;
    constexpr int iterations = 50;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> host_a(m * k);
    std::vector<float> host_b(k * n);
    std::vector<float> host_bias(n);
    for (float& value : host_a) {
        value = dist(rng);
    }
    for (float& value : host_b) {
        value = dist(rng);
    }
    for (float& value : host_bias) {
        value = dist(rng);
    }

    float* dev_a = nullptr;
    float* dev_b = nullptr;
    float* dev_bias = nullptr;
    float* dev_workspace = nullptr;
    float* dev_output = nullptr;
    CUDA_CHECK(cudaMalloc(&dev_a, host_a.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_b, host_b.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_bias, host_bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_workspace, m * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_output, m * n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev_a, host_a.data(), host_a.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b, host_b.data(), host_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_bias, host_bias.data(), host_bias.size() * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) {
        matmul_bias_gelu(dev_a, dev_b, dev_bias, dev_workspace, dev_output, m, n, k);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        matmul_bias_gelu(dev_a, dev_b, dev_bias, dev_workspace, dev_output, m, n, k);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::printf("matmul_bias_gelu m=%d n=%d k=%d avg_ms=%.4f\n",
                m, n, k, milliseconds / iterations);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_bias);
    cudaFree(dev_workspace);
    cudaFree(dev_output);
    return 0;
}
