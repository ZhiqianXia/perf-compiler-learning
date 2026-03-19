# Linear Algebra Operations

Matrix and vector operations fundamental to deep learning.

## Operators

### `matmul/` - Matrix Multiplication
- Input: A[M×K], B[K×N]
- Output: C[M×N] = A @ B
- Implementation: Tiled kernel for cache efficiency

### `gemm/` - General Matrix Multiplication
- ALPHA * A @ B + BETA * C
- Supports alpha/beta scaling

## Optimization Strategies

1. **Tiling**: 16×16 or 32×32 blocks in shared memory
2. **Coalescing**: Arrange warps to access contiguous memory
3. **Occupancy**: 2-4 blocks per multiprocessor
4. **Warps**: Common patterns for warp-level operations

## Performance Notes

- Peak bandwidth utilization: ~85-95%
- Register pressure: 50-100 per thread
- Shared memory: 16-32 KB per block
