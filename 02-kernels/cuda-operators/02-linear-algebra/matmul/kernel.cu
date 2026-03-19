#include <cuda_runtime.h>

// Naive MatMul kernel
__global__ void matmul_naive_kernel(const float* A, const float* B, float* C,
                                     int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled MatMul kernel (better cache efficiency)
#define TILE_SIZE 16

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C,
                                     int M, int N, int K) {
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tiles
        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;

        if (row < M && a_col < K) {
            A_tile[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (b_row < K && col < N) {
            B_tile[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            B_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute tile contribution
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matmul(const float* A, const float* B, float* C, int M, int N, int K,
            cudaStream_t stream = 0) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_tiled_kernel<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
}
