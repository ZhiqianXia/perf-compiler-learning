#include <cuda_runtime.h>

#include <cstdio>
#include <random>
#include <vector>

void conv2d_nchw(const float* input,
                 const float* weight,
                 const float* bias,
                 float* output,
                 int batch,
                 int in_channels,
                 int out_channels,
                 int in_h,
                 int in_w,
                 int kernel_h,
                 int kernel_w,
                 int stride_h,
                 int stride_w,
                 int pad_h,
                 int pad_w,
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
    constexpr int batch = 16;
    constexpr int in_channels = 32;
    constexpr int out_channels = 64;
    constexpr int in_h = 56;
    constexpr int in_w = 56;
    constexpr int kernel_h = 3;
    constexpr int kernel_w = 3;
    constexpr int stride = 1;
    constexpr int pad = 1;
    constexpr int out_h = (in_h + 2 * pad - kernel_h) / stride + 1;
    constexpr int out_w = (in_w + 2 * pad - kernel_w) / stride + 1;
    constexpr int warmup = 10;
    constexpr int iterations = 50;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> host_input(batch * in_channels * in_h * in_w);
    std::vector<float> host_weight(out_channels * in_channels * kernel_h * kernel_w);
    std::vector<float> host_bias(out_channels);
    for (float& value : host_input) {
        value = dist(rng);
    }
    for (float& value : host_weight) {
        value = dist(rng);
    }
    for (float& value : host_bias) {
        value = dist(rng);
    }

    float* dev_input = nullptr;
    float* dev_weight = nullptr;
    float* dev_bias = nullptr;
    float* dev_output = nullptr;
    CUDA_CHECK(cudaMalloc(&dev_input, host_input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_weight, host_weight.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_bias, host_bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_output,
                          batch * out_channels * out_h * out_w * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev_input,
                          host_input.data(),
                          host_input.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_weight,
                          host_weight.data(),
                          host_weight.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_bias,
                          host_bias.data(),
                          host_bias.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) {
        conv2d_nchw(dev_input,
                    dev_weight,
                    dev_bias,
                    dev_output,
                    batch,
                    in_channels,
                    out_channels,
                    in_h,
                    in_w,
                    kernel_h,
                    kernel_w,
                    stride,
                    stride,
                    pad,
                    pad);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        conv2d_nchw(dev_input,
                    dev_weight,
                    dev_bias,
                    dev_output,
                    batch,
                    in_channels,
                    out_channels,
                    in_h,
                    in_w,
                    kernel_h,
                    kernel_w,
                    stride,
                    stride,
                    pad,
                    pad);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::printf("conv2d nchw shape=(%d,%d,%d,%d) kernel=%dx%d avg_ms=%.4f\n",
                batch,
                in_channels,
                in_h,
                in_w,
                kernel_h,
                kernel_w,
                milliseconds / iterations);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_input);
    cudaFree(dev_weight);
    cudaFree(dev_bias);
    cudaFree(dev_output);
    return 0;
}
