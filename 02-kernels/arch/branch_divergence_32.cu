/**
 * branch_divergence_32.cu
 *
 * 32层嵌套分支的CUDA程序，用于研究warp分支发散(branch divergence)。
 *
 * 核心思路：
 *   - 一个warp有32个线程 (lane 0~31)
 *   - 每一层用 threadIdx.x 的第 i 位 (bit i) 做条件分支
 *   - 32层嵌套后，每个线程走的路径都不同 → 2^32 条唯一路径
 *   - 这是warp divergence的极端情况，方便用 nsight compute 观察
 *
 * 编译:
 *   nvcc -O2 -arch=sm_80 branch_divergence_32.cu -o branch_divergence_32
 *
 * Profiling:
 *   ncu --set full ./branch_divergence_32
 *   ncu --metrics smsp__branch_targets_serially_executed ./branch_divergence_32
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
// Kernel: 32层嵌套 if-else，每层按 bit i 分叉
// ============================================================
__global__ void branch_32_levels(uint32_t *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // lane: 线程在warp内的编号 (0~31)
    uint32_t lane = threadIdx.x & 31;
    uint32_t val  = 0;

    // ---------- Level 0 (bit 0) ----------
    if (lane & (1u << 0)) {
        val ^= 0x00000001u;
        // ---------- Level 1 (bit 1) ----------
        if (lane & (1u << 1)) {
            val ^= 0x00000003u;
            // ---------- Level 2 (bit 2) ----------
            if (lane & (1u << 2)) {
                val ^= 0x00000007u;
                // ---------- Level 3 (bit 3) ----------
                if (lane & (1u << 3)) {
                    val ^= 0x0000000Fu;
                    // ---------- Level 4 ----------
                    if (lane & (1u << 4)) {
                        val ^= 0x0000001Fu;
                        // ---------- Level 5 ----------
                        if (lane & (1u << 5)) {
                            val ^= 0x0000003Fu;
                            // ---------- Level 6 ----------
                            if (lane & (1u << 6)) {
                                val ^= 0x0000007Fu;
                                // ---------- Level 7 ----------
                                if (lane & (1u << 7)) {
                                    val ^= 0x000000FFu;
                                    // L8
                                    if (lane & (1u << 8)) { val ^= 0x000001FFu; } else { val += 0x100u; }
                                    // L9
                                    if (lane & (1u << 9)) { val ^= 0x000003FFu; } else { val += 0x200u; }
                                    // L10
                                    if (lane & (1u << 10)) { val ^= 0x000007FFu; } else { val += 0x400u; }
                                    // L11
                                    if (lane & (1u << 11)) { val ^= 0x00000FFFu; } else { val += 0x800u; }
                                    // L12
                                    if (lane & (1u << 12)) { val ^= 0x00001FFFu; } else { val += 0x1000u; }
                                    // L13
                                    if (lane & (1u << 13)) { val ^= 0x00003FFFu; } else { val += 0x2000u; }
                                    // L14
                                    if (lane & (1u << 14)) { val ^= 0x00007FFFu; } else { val += 0x4000u; }
                                    // L15
                                    if (lane & (1u << 15)) { val ^= 0x0000FFFFu; } else { val += 0x8000u; }
                                } else {
                                    val += 0x80u;
                                    if (lane & (1u << 8)) { val ^= 0x00000180u; } else { val += 0x100u; }
                                    if (lane & (1u << 9)) { val ^= 0x00000280u; } else { val += 0x200u; }
                                    if (lane & (1u << 10)) { val ^= 0x00000480u; } else { val += 0x400u; }
                                    if (lane & (1u << 11)) { val ^= 0x00000880u; } else { val += 0x800u; }
                                    if (lane & (1u << 12)) { val ^= 0x00001080u; } else { val += 0x1000u; }
                                    if (lane & (1u << 13)) { val ^= 0x00002080u; } else { val += 0x2000u; }
                                    if (lane & (1u << 14)) { val ^= 0x00004080u; } else { val += 0x4000u; }
                                    if (lane & (1u << 15)) { val ^= 0x00008080u; } else { val += 0x8000u; }
                                }
                            } else {
                                val += 0x40u;
                                if (lane & (1u << 7)) { val ^= 0xC0u; } else { val += 0x80u; }
                                if (lane & (1u << 8)) { val ^= 0x140u; } else { val += 0x100u; }
                                if (lane & (1u << 9)) { val ^= 0x240u; } else { val += 0x200u; }
                                if (lane & (1u << 10)) { val ^= 0x440u; } else { val += 0x400u; }
                                if (lane & (1u << 11)) { val ^= 0x840u; } else { val += 0x800u; }
                                if (lane & (1u << 12)) { val ^= 0x1040u; } else { val += 0x1000u; }
                                if (lane & (1u << 13)) { val ^= 0x2040u; } else { val += 0x2000u; }
                                if (lane & (1u << 14)) { val ^= 0x4040u; } else { val += 0x4000u; }
                                if (lane & (1u << 15)) { val ^= 0x8040u; } else { val += 0x8000u; }
                            }
                        } else {
                            val += 0x20u;
                            if (lane & (1u << 6)) { val ^= 0x60u; } else { val += 0x40u; }
                            if (lane & (1u << 7)) { val ^= 0xA0u; } else { val += 0x80u; }
                            if (lane & (1u << 8)) { val ^= 0x120u; } else { val += 0x100u; }
                            if (lane & (1u << 9)) { val ^= 0x220u; } else { val += 0x200u; }
                            if (lane & (1u << 10)) { val ^= 0x420u; } else { val += 0x400u; }
                            if (lane & (1u << 11)) { val ^= 0x820u; } else { val += 0x800u; }
                            if (lane & (1u << 12)) { val ^= 0x1020u; } else { val += 0x1000u; }
                            if (lane & (1u << 13)) { val ^= 0x2020u; } else { val += 0x2000u; }
                            if (lane & (1u << 14)) { val ^= 0x4020u; } else { val += 0x4000u; }
                            if (lane & (1u << 15)) { val ^= 0x8020u; } else { val += 0x8000u; }
                        }
                    } else {
                        val += 0x10u;
                        if (lane & (1u << 5)) { val ^= 0x30u; } else { val += 0x20u; }
                        if (lane & (1u << 6)) { val ^= 0x50u; } else { val += 0x40u; }
                        if (lane & (1u << 7)) { val ^= 0x90u; } else { val += 0x80u; }
                        if (lane & (1u << 8)) { val ^= 0x110u; } else { val += 0x100u; }
                        if (lane & (1u << 9)) { val ^= 0x210u; } else { val += 0x200u; }
                        if (lane & (1u << 10)) { val ^= 0x410u; } else { val += 0x400u; }
                        if (lane & (1u << 11)) { val ^= 0x810u; } else { val += 0x800u; }
                        if (lane & (1u << 12)) { val ^= 0x1010u; } else { val += 0x1000u; }
                        if (lane & (1u << 13)) { val ^= 0x2010u; } else { val += 0x2000u; }
                        if (lane & (1u << 14)) { val ^= 0x4010u; } else { val += 0x4000u; }
                        if (lane & (1u << 15)) { val ^= 0x8010u; } else { val += 0x8000u; }
                    }
                } else {
                    val += 0x08u;
                    if (lane & (1u << 4)) { val ^= 0x18u; } else { val += 0x10u; }
                    if (lane & (1u << 5)) { val ^= 0x28u; } else { val += 0x20u; }
                    if (lane & (1u << 6)) { val ^= 0x48u; } else { val += 0x40u; }
                    if (lane & (1u << 7)) { val ^= 0x88u; } else { val += 0x80u; }
                    if (lane & (1u << 8)) { val ^= 0x108u; } else { val += 0x100u; }
                    if (lane & (1u << 9)) { val ^= 0x208u; } else { val += 0x200u; }
                    if (lane & (1u << 10)) { val ^= 0x408u; } else { val += 0x400u; }
                    if (lane & (1u << 11)) { val ^= 0x808u; } else { val += 0x800u; }
                    if (lane & (1u << 12)) { val ^= 0x1008u; } else { val += 0x1000u; }
                    if (lane & (1u << 13)) { val ^= 0x2008u; } else { val += 0x2000u; }
                    if (lane & (1u << 14)) { val ^= 0x4008u; } else { val += 0x4000u; }
                    if (lane & (1u << 15)) { val ^= 0x8008u; } else { val += 0x8000u; }
                }
            } else {
                val += 0x04u;
                if (lane & (1u << 3)) { val ^= 0x0Cu; } else { val += 0x08u; }
                if (lane & (1u << 4)) { val ^= 0x14u; } else { val += 0x10u; }
                if (lane & (1u << 5)) { val ^= 0x24u; } else { val += 0x20u; }
                if (lane & (1u << 6)) { val ^= 0x44u; } else { val += 0x40u; }
                if (lane & (1u << 7)) { val ^= 0x84u; } else { val += 0x80u; }
                if (lane & (1u << 8)) { val ^= 0x104u; } else { val += 0x100u; }
                if (lane & (1u << 9)) { val ^= 0x204u; } else { val += 0x200u; }
                if (lane & (1u << 10)) { val ^= 0x404u; } else { val += 0x400u; }
                if (lane & (1u << 11)) { val ^= 0x804u; } else { val += 0x800u; }
                if (lane & (1u << 12)) { val ^= 0x1004u; } else { val += 0x1000u; }
                if (lane & (1u << 13)) { val ^= 0x2004u; } else { val += 0x2000u; }
                if (lane & (1u << 14)) { val ^= 0x4004u; } else { val += 0x4000u; }
                if (lane & (1u << 15)) { val ^= 0x8004u; } else { val += 0x8000u; }
            }
        } else {
            val += 0x02u;
            if (lane & (1u << 2)) { val ^= 0x06u; } else { val += 0x04u; }
            if (lane & (1u << 3)) { val ^= 0x0Au; } else { val += 0x08u; }
            if (lane & (1u << 4)) { val ^= 0x12u; } else { val += 0x10u; }
            if (lane & (1u << 5)) { val ^= 0x22u; } else { val += 0x20u; }
            if (lane & (1u << 6)) { val ^= 0x42u; } else { val += 0x40u; }
            if (lane & (1u << 7)) { val ^= 0x82u; } else { val += 0x80u; }
            if (lane & (1u << 8)) { val ^= 0x102u; } else { val += 0x100u; }
            if (lane & (1u << 9)) { val ^= 0x202u; } else { val += 0x200u; }
            if (lane & (1u << 10)) { val ^= 0x402u; } else { val += 0x400u; }
            if (lane & (1u << 11)) { val ^= 0x802u; } else { val += 0x800u; }
            if (lane & (1u << 12)) { val ^= 0x1002u; } else { val += 0x1000u; }
            if (lane & (1u << 13)) { val ^= 0x2002u; } else { val += 0x2000u; }
            if (lane & (1u << 14)) { val ^= 0x4002u; } else { val += 0x4000u; }
            if (lane & (1u << 15)) { val ^= 0x8002u; } else { val += 0x8000u; }
        }
    } else {
        val += 0x01u;
        // mirror: levels 1~15 on the else side
        if (lane & (1u << 1)) { val ^= 0x03u; } else { val += 0x02u; }
        if (lane & (1u << 2)) { val ^= 0x05u; } else { val += 0x04u; }
        if (lane & (1u << 3)) { val ^= 0x09u; } else { val += 0x08u; }
        if (lane & (1u << 4)) { val ^= 0x11u; } else { val += 0x10u; }
        if (lane & (1u << 5)) { val ^= 0x21u; } else { val += 0x20u; }
        if (lane & (1u << 6)) { val ^= 0x41u; } else { val += 0x40u; }
        if (lane & (1u << 7)) { val ^= 0x81u; } else { val += 0x80u; }
        if (lane & (1u << 8)) { val ^= 0x101u; } else { val += 0x100u; }
        if (lane & (1u << 9)) { val ^= 0x201u; } else { val += 0x200u; }
        if (lane & (1u << 10)) { val ^= 0x401u; } else { val += 0x400u; }
        if (lane & (1u << 11)) { val ^= 0x801u; } else { val += 0x800u; }
        if (lane & (1u << 12)) { val ^= 0x1001u; } else { val += 0x1000u; }
        if (lane & (1u << 13)) { val ^= 0x2001u; } else { val += 0x2000u; }
        if (lane & (1u << 14)) { val ^= 0x4001u; } else { val += 0x4000u; }
        if (lane & (1u << 15)) { val ^= 0x8001u; } else { val += 0x8000u; }
    }

    // ---- Levels 16~31: 第二轮16层嵌套 ----
    // 用另一种方式产生分支：按 bit 16~31 进行嵌套分叉
    // 因为 lane 只有 0~31 (5 bits)，bit 5~31 对 lane 来说都是 0
    // 所以这里用 tid 的高位来驱动更多分支
    uint32_t key = (uint32_t)tid;

    // Level 16
    if (key & (1u << 0)) {
        val ^= 0x00010000u;
        // Level 17
        if (key & (1u << 1)) {
            val ^= 0x00030000u;
            // Level 18
            if (key & (1u << 2)) {
                val ^= 0x00070000u;
                // Level 19
                if (key & (1u << 3)) {
                    val ^= 0x000F0000u;
                    // Level 20
                    if (key & (1u << 4)) {
                        val ^= 0x001F0000u;
                        // Level 21
                        if (key & (1u << 5)) {
                            val ^= 0x003F0000u;
                            // Level 22
                            if (key & (1u << 6)) {
                                val ^= 0x007F0000u;
                                // Level 23
                                if (key & (1u << 7)) {
                                    val ^= 0x00FF0000u;
                                    // Levels 24~31
                                    if (key & (1u << 8))  { val ^= 0x01FF0000u; } else { val += 0x01000000u; }
                                    if (key & (1u << 9))  { val ^= 0x03FF0000u; } else { val += 0x02000000u; }
                                    if (key & (1u << 10)) { val ^= 0x07FF0000u; } else { val += 0x04000000u; }
                                    if (key & (1u << 11)) { val ^= 0x0FFF0000u; } else { val += 0x08000000u; }
                                    if (key & (1u << 12)) { val ^= 0x1FFF0000u; } else { val += 0x10000000u; }
                                    if (key & (1u << 13)) { val ^= 0x3FFF0000u; } else { val += 0x20000000u; }
                                    if (key & (1u << 14)) { val ^= 0x7FFF0000u; } else { val += 0x40000000u; }
                                    if (key & (1u << 15)) { val ^= 0xFFFF0000u; } else { val += 0x80000000u; }
                                } else {
                                    val += 0x00800000u;
                                    if (key & (1u << 8))  { val ^= 0x01800000u; } else { val += 0x01000000u; }
                                    if (key & (1u << 9))  { val ^= 0x02800000u; } else { val += 0x02000000u; }
                                    if (key & (1u << 10)) { val ^= 0x04800000u; } else { val += 0x04000000u; }
                                    if (key & (1u << 11)) { val ^= 0x08800000u; } else { val += 0x08000000u; }
                                    if (key & (1u << 12)) { val ^= 0x10800000u; } else { val += 0x10000000u; }
                                    if (key & (1u << 13)) { val ^= 0x20800000u; } else { val += 0x20000000u; }
                                    if (key & (1u << 14)) { val ^= 0x40800000u; } else { val += 0x40000000u; }
                                    if (key & (1u << 15)) { val ^= 0x80800000u; } else { val += 0x80000000u; }
                                }
                            } else {
                                val += 0x00400000u;
                                if (key & (1u << 7))  { val ^= 0x00C00000u; } else { val += 0x00800000u; }
                                if (key & (1u << 8))  { val ^= 0x01400000u; } else { val += 0x01000000u; }
                                if (key & (1u << 9))  { val ^= 0x02400000u; } else { val += 0x02000000u; }
                                if (key & (1u << 10)) { val ^= 0x04400000u; } else { val += 0x04000000u; }
                                if (key & (1u << 11)) { val ^= 0x08400000u; } else { val += 0x08000000u; }
                                if (key & (1u << 12)) { val ^= 0x10400000u; } else { val += 0x10000000u; }
                                if (key & (1u << 13)) { val ^= 0x20400000u; } else { val += 0x20000000u; }
                                if (key & (1u << 14)) { val ^= 0x40400000u; } else { val += 0x40000000u; }
                                if (key & (1u << 15)) { val ^= 0x80400000u; } else { val += 0x80000000u; }
                            }
                        } else {
                            val += 0x00200000u;
                            if (key & (1u << 6))  { val ^= 0x00600000u; } else { val += 0x00400000u; }
                            if (key & (1u << 7))  { val ^= 0x00A00000u; } else { val += 0x00800000u; }
                            if (key & (1u << 8))  { val ^= 0x01200000u; } else { val += 0x01000000u; }
                            if (key & (1u << 9))  { val ^= 0x02200000u; } else { val += 0x02000000u; }
                            if (key & (1u << 10)) { val ^= 0x04200000u; } else { val += 0x04000000u; }
                            if (key & (1u << 11)) { val ^= 0x08200000u; } else { val += 0x08000000u; }
                            if (key & (1u << 12)) { val ^= 0x10200000u; } else { val += 0x10000000u; }
                            if (key & (1u << 13)) { val ^= 0x20200000u; } else { val += 0x20000000u; }
                            if (key & (1u << 14)) { val ^= 0x40200000u; } else { val += 0x40000000u; }
                            if (key & (1u << 15)) { val ^= 0x80200000u; } else { val += 0x80000000u; }
                        }
                    } else {
                        val += 0x00100000u;
                        if (key & (1u << 5)) { val ^= 0x00300000u; } else { val += 0x00200000u; }
                        if (key & (1u << 6)) { val ^= 0x00500000u; } else { val += 0x00400000u; }
                        if (key & (1u << 7)) { val ^= 0x00900000u; } else { val += 0x00800000u; }
                        if (key & (1u << 8))  { val ^= 0x01100000u; } else { val += 0x01000000u; }
                        if (key & (1u << 9))  { val ^= 0x02100000u; } else { val += 0x02000000u; }
                        if (key & (1u << 10)) { val ^= 0x04100000u; } else { val += 0x04000000u; }
                        if (key & (1u << 11)) { val ^= 0x08100000u; } else { val += 0x08000000u; }
                        if (key & (1u << 12)) { val ^= 0x10100000u; } else { val += 0x10000000u; }
                        if (key & (1u << 13)) { val ^= 0x20100000u; } else { val += 0x20000000u; }
                        if (key & (1u << 14)) { val ^= 0x40100000u; } else { val += 0x40000000u; }
                        if (key & (1u << 15)) { val ^= 0x80100000u; } else { val += 0x80000000u; }
                    }
                } else {
                    val += 0x00080000u;
                    if (key & (1u << 4)) { val ^= 0x00180000u; } else { val += 0x00100000u; }
                    if (key & (1u << 5)) { val ^= 0x00280000u; } else { val += 0x00200000u; }
                    if (key & (1u << 6)) { val ^= 0x00480000u; } else { val += 0x00400000u; }
                    if (key & (1u << 7)) { val ^= 0x00880000u; } else { val += 0x00800000u; }
                    if (key & (1u << 8))  { val ^= 0x01080000u; } else { val += 0x01000000u; }
                    if (key & (1u << 9))  { val ^= 0x02080000u; } else { val += 0x02000000u; }
                    if (key & (1u << 10)) { val ^= 0x04080000u; } else { val += 0x04000000u; }
                    if (key & (1u << 11)) { val ^= 0x08080000u; } else { val += 0x08000000u; }
                    if (key & (1u << 12)) { val ^= 0x10080000u; } else { val += 0x10000000u; }
                    if (key & (1u << 13)) { val ^= 0x20080000u; } else { val += 0x20000000u; }
                    if (key & (1u << 14)) { val ^= 0x40080000u; } else { val += 0x40000000u; }
                    if (key & (1u << 15)) { val ^= 0x80080000u; } else { val += 0x80000000u; }
                }
            } else {
                val += 0x00040000u;
                if (key & (1u << 3)) { val ^= 0x000C0000u; } else { val += 0x00080000u; }
                if (key & (1u << 4)) { val ^= 0x00140000u; } else { val += 0x00100000u; }
                if (key & (1u << 5)) { val ^= 0x00240000u; } else { val += 0x00200000u; }
                if (key & (1u << 6)) { val ^= 0x00440000u; } else { val += 0x00400000u; }
                if (key & (1u << 7)) { val ^= 0x00840000u; } else { val += 0x00800000u; }
                if (key & (1u << 8))  { val ^= 0x01040000u; } else { val += 0x01000000u; }
                if (key & (1u << 9))  { val ^= 0x02040000u; } else { val += 0x02000000u; }
                if (key & (1u << 10)) { val ^= 0x04040000u; } else { val += 0x04000000u; }
                if (key & (1u << 11)) { val ^= 0x08040000u; } else { val += 0x08000000u; }
                if (key & (1u << 12)) { val ^= 0x10040000u; } else { val += 0x10000000u; }
                if (key & (1u << 13)) { val ^= 0x20040000u; } else { val += 0x20000000u; }
                if (key & (1u << 14)) { val ^= 0x40040000u; } else { val += 0x40000000u; }
                if (key & (1u << 15)) { val ^= 0x80040000u; } else { val += 0x80000000u; }
            }
        } else {
            val += 0x00020000u;
            if (key & (1u << 2)) { val ^= 0x00060000u; } else { val += 0x00040000u; }
            if (key & (1u << 3)) { val ^= 0x000A0000u; } else { val += 0x00080000u; }
            if (key & (1u << 4)) { val ^= 0x00120000u; } else { val += 0x00100000u; }
            if (key & (1u << 5)) { val ^= 0x00220000u; } else { val += 0x00200000u; }
            if (key & (1u << 6)) { val ^= 0x00420000u; } else { val += 0x00400000u; }
            if (key & (1u << 7)) { val ^= 0x00820000u; } else { val += 0x00800000u; }
            if (key & (1u << 8))  { val ^= 0x01020000u; } else { val += 0x01000000u; }
            if (key & (1u << 9))  { val ^= 0x02020000u; } else { val += 0x02000000u; }
            if (key & (1u << 10)) { val ^= 0x04020000u; } else { val += 0x04000000u; }
            if (key & (1u << 11)) { val ^= 0x08020000u; } else { val += 0x08000000u; }
            if (key & (1u << 12)) { val ^= 0x10020000u; } else { val += 0x10000000u; }
            if (key & (1u << 13)) { val ^= 0x20020000u; } else { val += 0x20000000u; }
            if (key & (1u << 14)) { val ^= 0x40020000u; } else { val += 0x40000000u; }
            if (key & (1u << 15)) { val ^= 0x80020000u; } else { val += 0x80000000u; }
        }
    } else {
        val += 0x00010000u;
        if (key & (1u << 1))  { val ^= 0x00030000u; } else { val += 0x00020000u; }
        if (key & (1u << 2))  { val ^= 0x00050000u; } else { val += 0x00040000u; }
        if (key & (1u << 3))  { val ^= 0x00090000u; } else { val += 0x00080000u; }
        if (key & (1u << 4))  { val ^= 0x00110000u; } else { val += 0x00100000u; }
        if (key & (1u << 5))  { val ^= 0x00210000u; } else { val += 0x00200000u; }
        if (key & (1u << 6))  { val ^= 0x00410000u; } else { val += 0x00400000u; }
        if (key & (1u << 7))  { val ^= 0x00810000u; } else { val += 0x00800000u; }
        if (key & (1u << 8))  { val ^= 0x01010000u; } else { val += 0x01000000u; }
        if (key & (1u << 9))  { val ^= 0x02010000u; } else { val += 0x02000000u; }
        if (key & (1u << 10)) { val ^= 0x04010000u; } else { val += 0x04000000u; }
        if (key & (1u << 11)) { val ^= 0x08010000u; } else { val += 0x08000000u; }
        if (key & (1u << 12)) { val ^= 0x10010000u; } else { val += 0x10000000u; }
        if (key & (1u << 13)) { val ^= 0x20010000u; } else { val += 0x20000000u; }
        if (key & (1u << 14)) { val ^= 0x40010000u; } else { val += 0x40000000u; }
        if (key & (1u << 15)) { val ^= 0x80010000u; } else { val += 0x80000000u; }
    }

    output[tid] = val;
}

// ============================================================
// Kernel: 无分支的对照组 (baseline)
// ============================================================
__global__ void no_branch_baseline(uint32_t *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    uint32_t lane = threadIdx.x & 31;
    uint32_t val = lane * 137u + 42u;

    // 32次简单运算，无分支
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        val = val ^ (val << 1) ^ (uint32_t)i;
    }

    output[tid] = val;
}

// ============================================================
// Kernel: 使用循环+位测试产生32层分支 (更紧凑的写法)
// 编译器可能会展开，但逻辑上是32层分叉
// ============================================================
__global__ void branch_32_loop(uint32_t *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    uint32_t lane = threadIdx.x & 31;
    uint32_t val  = 0;

    // 用volatile阻止编译器优化掉分支
    volatile uint32_t bits = lane ^ (uint32_t)tid;

    #pragma unroll
    for (int i = 0; i < 32; i++) {
        if (bits & (1u << i)) {
            val += (1u << i);           // if 路径
            val ^= ((uint32_t)i * 7u);
        } else {
            val -= (1u << i);           // else 路径
            val ^= ((uint32_t)i * 13u);
        }
    }

    output[tid] = val;
}

// ============================================================
// main
// ============================================================
int main() {
    const int N = 1024;  // 32 warps = 1024 threads
    const int BLOCK = 256;
    const int GRID  = (N + BLOCK - 1) / BLOCK;

    uint32_t *d_out = nullptr;
    uint32_t *h_out = (uint32_t *)malloc(N * sizeof(uint32_t));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(uint32_t)));

    // ---- 1. 32层嵌套分支 ----
    printf("=== Kernel 1: 32-level nested branch divergence ===\n");
    CHECK_CUDA(cudaMemset(d_out, 0, N * sizeof(uint32_t)));
    branch_32_levels<<<GRID, BLOCK>>>(d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    printf("First warp results (thread 0~31):\n");
    for (int i = 0; i < 32; i++) {
        printf("  lane %2d: 0x%08X\n", i, h_out[i]);
    }

    // 验证: 每个lane的结果应该不同
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

    // ---- 3. 循环式32层分支 ----
    printf("=== Kernel 3: 32-level loop-based divergence ===\n");
    CHECK_CUDA(cudaMemset(d_out, 0, N * sizeof(uint32_t)));
    branch_32_loop<<<GRID, BLOCK>>>(d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    printf("First warp results (thread 0~31):\n");
    for (int i = 0; i < 32; i++) {
        printf("  lane %2d: 0x%08X\n", i, h_out[i]);
    }

    // ---- Timing comparison ----
    printf("\n=== Timing (100 iterations) ===\n");
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float ms;

    // Divergent kernel
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++)
        branch_32_levels<<<GRID, BLOCK>>>(d_out, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("branch_32_levels : %.3f ms\n", ms);

    // Baseline
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++)
        no_branch_baseline<<<GRID, BLOCK>>>(d_out, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("no_branch_baseline: %.3f ms\n", ms);

    // Loop-based divergent
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++)
        branch_32_loop<<<GRID, BLOCK>>>(d_out, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("branch_32_loop   : %.3f ms\n", ms);

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_out));
    free(h_out);

    printf("\nDone. Use 'ncu --set full ./branch_divergence_32' to profile.\n");
    return 0;
}
