#include <cuda_runtime.h>

#include <cstdio>
#include <random>
#include <vector>

void matmul(const float* A, const float* B, float* C, int M, int N, int K,
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
    constexpr int M = 512;
    constexpr int N = 512;
    constexpr int K = 512;
    constexpr int warmup = 10;
    constexpr int iterations = 100;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> host_a(M * K);
    std::vector<float> host_b(K * N);
    for (float& value : host_a) {
        value = dist(rng);
    }
    for (float& value : host_b) {
        value = dist(rng);
    }

    float* dev_a = nullptr;
    float* dev_b = nullptr;
    float* dev_c = nullptr;
    CUDA_CHECK(cudaMalloc(&dev_a, host_a.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_b, host_b.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_c, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev_a, host_a.data(), host_a.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b, host_b.data(), host_b.size() * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) {
        matmul(dev_a, dev_b, dev_c, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        matmul(dev_a, dev_b, dev_c, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    const double flops = 2.0 * static_cast<double>(M) * N * K;
    const double tflops = (flops / (milliseconds / iterations) * 1e-9) / 1e3;
    std::printf("matmul shape=(%d,%d)x(%d,%d) avg_ms=%.4f approx_tflops=%.3f\n",
                M, K, K, N, milliseconds / iterations, tflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}
