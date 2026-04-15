/**
 * branch_divergence_8.cu
 *
 * 8层嵌套分支的CUDA程序，用于研究warp分支发散(branch divergence)。
 *
 * 核心思路：
 *   - 用 lane (threadIdx.x & 31) 的 bit 0~7 做8层嵌套 if-else
 *   - 同一 warp 内线程走不同路径 → 分支发散
 *   - 包含无分支对照组和循环式分支版本
 *
 * 编译:
 *   nvcc -O2 -arch=sm_80 branch_divergence_8.cu -o branch_divergence_8
 *
 * Profiling:
 *   ncu --set full ./branch_divergence_8
 */

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ============================================================
// Kernel 1: 8层嵌套 if-else，每层按 bit i 分叉
// ============================================================
__global__ void branch_8_nested(uint32_t *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    uint32_t lane = threadIdx.x & 31;
    uint32_t val  = 0;

    // Level 0 (bit 0)
    if (lane & (1u << 0)) {
        val ^= 0x01u;
        // Level 1 (bit 1)
        if (lane & (1u << 1)) {
            val ^= 0x03u;
            // Level 2 (bit 2)
            if (lane & (1u << 2)) {
                val ^= 0x07u;
                // Level 3 (bit 3)
                if (lane & (1u << 3)) {
                    val ^= 0x0Fu;
                    // Level 4 (bit 4)
                    if (lane & (1u << 4)) {
                        val ^= 0x1Fu;
                        // Level 5 (bit 5)
                        if (lane & (1u << 5)) {
                            val ^= 0x3Fu;
                            // Level 6 (bit 6)
                            if (lane & (1u << 6)) {
                                val ^= 0x7Fu;
                                // Level 7 (bit 7)
                                if (lane & (1u << 7)) {
                                    val ^= 0xFFu;
                                } else {
                                    val += 0x80u;
                                }
                            } else {
                                val += 0x40u;
                                if (lane & (1u << 7)) {
                                    val ^= 0xC0u;
                                } else {
                                    val += 0x80u;
                                }
                            }
                        } else {
                            val += 0x20u;
                            if (lane & (1u << 6)) {
                                val ^= 0x60u;
                            } else {
                                val += 0x40u;
                            }
                            if (lane & (1u << 7)) {
                                val ^= 0xA0u;
                            } else {
                                val += 0x80u;
                            }
                        }
                    } else {
                        val += 0x10u;
                        if (lane & (1u << 5)) { val ^= 0x30u; } else { val += 0x20u; }
                        if (lane & (1u << 6)) { val ^= 0x50u; } else { val += 0x40u; }
                        if (lane & (1u << 7)) { val ^= 0x90u; } else { val += 0x80u; }
                    }
                } else {
                    val += 0x08u;
                    if (lane & (1u << 4)) { val ^= 0x18u; } else { val += 0x10u; }
                    if (lane & (1u << 5)) { val ^= 0x28u; } else { val += 0x20u; }
                    if (lane & (1u << 6)) { val ^= 0x48u; } else { val += 0x40u; }
                    if (lane & (1u << 7)) { val ^= 0x88u; } else { val += 0x80u; }
                }
            } else {
                val += 0x04u;
                if (lane & (1u << 3)) { val ^= 0x0Cu; } else { val += 0x08u; }
                if (lane & (1u << 4)) { val ^= 0x14u; } else { val += 0x10u; }
                if (lane & (1u << 5)) { val ^= 0x24u; } else { val += 0x20u; }
                if (lane & (1u << 6)) { val ^= 0x44u; } else { val += 0x40u; }
                if (lane & (1u << 7)) { val ^= 0x84u; } else { val += 0x80u; }
            }
        } else {
            val += 0x02u;
            if (lane & (1u << 2)) { val ^= 0x06u; } else { val += 0x04u; }
            if (lane & (1u << 3)) { val ^= 0x0Au; } else { val += 0x08u; }
            if (lane & (1u << 4)) { val ^= 0x12u; } else { val += 0x10u; }
            if (lane & (1u << 5)) { val ^= 0x22u; } else { val += 0x20u; }
            if (lane & (1u << 6)) { val ^= 0x42u; } else { val += 0x40u; }
            if (lane & (1u << 7)) { val ^= 0x82u; } else { val += 0x80u; }
        }
    } else {
        val += 0x01u;
        if (lane & (1u << 1)) { val ^= 0x03u; } else { val += 0x02u; }
        if (lane & (1u << 2)) { val ^= 0x05u; } else { val += 0x04u; }
        if (lane & (1u << 3)) { val ^= 0x09u; } else { val += 0x08u; }
        if (lane & (1u << 4)) { val ^= 0x11u; } else { val += 0x10u; }
        if (lane & (1u << 5)) { val ^= 0x21u; } else { val += 0x20u; }
        if (lane & (1u << 6)) { val ^= 0x41u; } else { val += 0x40u; }
        if (lane & (1u << 7)) { val ^= 0x81u; } else { val += 0x80u; }
    }

    output[tid] = val;
}

// ============================================================
// Kernel 2: 无分支对照组
// ============================================================
__global__ void no_branch_baseline(uint32_t *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    uint32_t lane = threadIdx.x & 31;
    uint32_t val = lane * 137u + 42u;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        val = val ^ (val << 1) ^ (uint32_t)i;
    }

    output[tid] = val;
}

// ============================================================
// Kernel 3: 循环式8层分支 (紧凑写法)
// ============================================================
__global__ void branch_8_loop(uint32_t *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    uint32_t lane = threadIdx.x & 31;
    uint32_t val  = 0;

    // volatile 阻止编译器优化掉分支
    volatile uint32_t bits = lane;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (bits & (1u << i)) {
            val += (1u << i);
            val ^= ((uint32_t)i * 7u);
        } else {
            val -= (1u << i);
            val ^= ((uint32_t)i * 13u);
        }
    }

    output[tid] = val;
}

// ============================================================
// main
// ============================================================
int main() {
    const int N = 1024;
    const int BLOCK = 256;
    const int GRID  = (N + BLOCK - 1) / BLOCK;

    uint32_t *d_out = nullptr;
    uint32_t *h_out = (uint32_t *)malloc(N * sizeof(uint32_t));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(uint32_t)));

    // ---- 1. 8层嵌套分支 ----
    printf("=== Kernel 1: 8-level nested branch divergence ===\n");
    CHECK_CUDA(cudaMemset(d_out, 0, N * sizeof(uint32_t)));
    branch_8_nested<<<GRID, BLOCK>>>(d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    printf("First warp results (thread 0~31):\n");
    for (int i = 0; i < 32; i++) {
        printf("  lane %2d: 0x%08X\n", i, h_out[i]);
    }

    int unique = 0;
    for (int i = 0; i < 32; i++) {
        int is_unique = 1;
        for (int j = 0; j < i; j++) {
            if (h_out[i] == h_out[j]) { is_unique = 0; break; }
        }
        unique += is_unique;
    }
    printf("Unique values in first warp: %d / 32\n\n", unique);

    // ---- 2. 无分支基准 ----
    printf("=== Kernel 2: No-branch baseline ===\n");
    CHECK_CUDA(cudaMemset(d_out, 0, N * sizeof(uint32_t)));
    no_branch_baseline<<<GRID, BLOCK>>>(d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    printf("lane 0: 0x%08X, lane 31: 0x%08X\n\n", h_out[0], h_out[31]);

    // ---- 3. 循环式8层分支 ----
    printf("=== Kernel 3: 8-level loop-based divergence ===\n");
    CHECK_CUDA(cudaMemset(d_out, 0, N * sizeof(uint32_t)));
    branch_8_loop<<<GRID, BLOCK>>>(d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    printf("First warp results (thread 0~31):\n");
    for (int i = 0; i < 32; i++) {
        printf("  lane %2d: 0x%08X\n", i, h_out[i]);
    }

    // ---- Timing ----
    printf("\n=== Timing (100 iterations) ===\n");
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float ms;

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++)
        branch_8_nested<<<GRID, BLOCK>>>(d_out, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("branch_8_nested  : %.3f ms\n", ms);

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++)
        no_branch_baseline<<<GRID, BLOCK>>>(d_out, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("no_branch_baseline: %.3f ms\n", ms);

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++)
        branch_8_loop<<<GRID, BLOCK>>>(d_out, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("branch_8_loop    : %.3f ms\n", ms);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_out));
    free(h_out);

    printf("\nDone. Use 'ncu --set full ./branch_divergence_8' to profile.\n");
    return 0;
}
