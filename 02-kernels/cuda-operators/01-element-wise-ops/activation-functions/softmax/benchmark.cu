#include <cuda_runtime.h>

#include <cstdio>
#include <random>
#include <vector>

void softmax(const float* x, float* y, int n, cudaStream_t stream = 0);

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
    constexpr int n = 4096;
    constexpr int warmup = 20;
    constexpr int iterations = 200;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<float> host_x(n);
    for (float& value : host_x) {
        value = dist(rng);
    }

    float* dev_x = nullptr;
    float* dev_y = nullptr;
    CUDA_CHECK(cudaMalloc(&dev_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_y, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev_x, host_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) {
        softmax(dev_x, dev_y, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        softmax(dev_x, dev_y, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::printf("softmax n=%d avg_ms=%.4f\n", n, milliseconds / iterations);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_x);
    cudaFree(dev_y);
    return 0;
}
