# MatMul Operation

## Description
Matrix multiplication: `C = A @ B`
- A: (M, K), B: (K, N) => C: (M, N)

## Kernel Implementation Strategy
- Tiling for cache efficiency
- Shared memory tile size: 16x16 or 32x32
- Coalesced memory access patterns
- WARP-level optimizations

## Performance Targets
- Memory bandwidth: >95% for large matrices
- Register usage: ~50-100 per thread

## Files
- `kernel.cu` - Optimized CUDA kernel with tiling
- `reference.cpp` - Host-side reference implementation for correctness checks
- `benchmark.cu` - Baseline benchmark entry for throughput measurement

## Validation Focus
- Tiny case: `M=4, N=5, K=3` for manual inspection
- Medium case: `512 x 512 x 512` for steady-state timing
- Edge case: dimensions not divisible by `TILE_SIZE`
