#include <cuda_runtime.h>

#include <cstdio>
#include <random>
#include <vector>

void layer_norm(const float* x, const float* gamma, const float* beta, float* y,
                int batch_size, int hidden_size, float eps = 1e-5f,
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
    constexpr int batch_size = 256;
    constexpr int hidden_size = 768;
    constexpr float eps = 1e-5f;
    constexpr int warmup = 20;
    constexpr int iterations = 100;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> host_x(batch_size * hidden_size);
    std::vector<float> host_gamma(hidden_size);
    std::vector<float> host_beta(hidden_size);
    for (float& value : host_x) {
        value = dist(rng);
    }
    for (float& value : host_gamma) {
        value = dist(rng);
    }
    for (float& value : host_beta) {
        value = dist(rng);
    }

    float* dev_x = nullptr;
    float* dev_gamma = nullptr;
    float* dev_beta = nullptr;
    float* dev_y = nullptr;
    CUDA_CHECK(cudaMalloc(&dev_x, host_x.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_gamma, host_gamma.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_beta, host_beta.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_y, host_x.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev_x, host_x.data(), host_x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_gamma, host_gamma.data(), host_gamma.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_beta, host_beta.data(), host_beta.size() * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) {
        layer_norm(dev_x, dev_gamma, dev_beta, dev_y, batch_size, hidden_size, eps);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        layer_norm(dev_x, dev_gamma, dev_beta, dev_y, batch_size, hidden_size, eps);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::printf("layer_norm shape=(%d,%d) avg_ms=%.4f\n",
                batch_size, hidden_size, milliseconds / iterations);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_x);
    cudaFree(dev_gamma);
    cudaFree(dev_beta);
    cudaFree(dev_y);
    return 0;
}
