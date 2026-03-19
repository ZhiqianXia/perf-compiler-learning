#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid bank conflicts

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row < rows && col < cols) {
        tile[threadIdx.y][threadIdx.x] = input[row * cols + col];
    }

    __syncthreads();

    int out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
    int out_col = blockIdx.y * TILE_SIZE + threadIdx.x;

    if (out_row < cols && out_col < rows) {
        output[out_row * rows + out_col] = tile[threadIdx.x][threadIdx.y];
    }
}

void transpose(const float* input, float* output, int rows, int cols,
               cudaStream_t stream = 0) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((cols + TILE_SIZE - 1) / TILE_SIZE,
                 (rows + TILE_SIZE - 1) / TILE_SIZE);
    transpose_kernel<<<gridDim, blockDim, 0, stream>>>(input, output, rows, cols);
}
