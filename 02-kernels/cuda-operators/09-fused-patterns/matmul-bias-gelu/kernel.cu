#include <cuda_runtime.h>

#include <cmath>

namespace {

constexpr int kTileSize = 16;

__device__ inline float gelu_approx(float x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
}

__global__ void matmul_tiled_kernel(const float* a,
                                    const float* b,
                                    float* c,
                                    int m,
                                    int n,
                                    int k) {
    __shared__ float a_tile[kTileSize][kTileSize];
    __shared__ float b_tile[kTileSize][kTileSize];

    const int row = blockIdx.y * kTileSize + threadIdx.y;
    const int col = blockIdx.x * kTileSize + threadIdx.x;
    float sum = 0.0f;

    for (int tile = 0; tile < (k + kTileSize - 1) / kTileSize; ++tile) {
        const int a_col = tile * kTileSize + threadIdx.x;
        const int b_row = tile * kTileSize + threadIdx.y;

        a_tile[threadIdx.y][threadIdx.x] =
            (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
        b_tile[threadIdx.y][threadIdx.x] =
            (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
        __syncthreads();

        for (int inner = 0; inner < kTileSize; ++inner) {
            sum += a_tile[threadIdx.y][inner] * b_tile[inner][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

__global__ void bias_gelu_kernel(const float* input,
                                 const float* bias,
                                 float* output,
                                 int m,
                                 int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = m * n;
    if (index >= total) {
        return;
    }

    const int col = index % n;
    output[index] = gelu_approx(input[index] + bias[col]);
}

}  // namespace

void matmul_bias_gelu(const float* a,
                      const float* b,
                      const float* bias,
                      float* workspace,
                      float* output,
                      int m,
                      int n,
                      int k,
                      cudaStream_t stream = 0) {
    dim3 block_dim(kTileSize, kTileSize);
    dim3 grid_dim((n + kTileSize - 1) / kTileSize, (m + kTileSize - 1) / kTileSize);
    matmul_tiled_kernel<<<grid_dim, block_dim, 0, stream>>>(a, b, workspace, m, n, k);

    const int total = m * n;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    bias_gelu_kernel<<<blocks, threads, 0, stream>>>(workspace, bias, output, m, n);
}
