# Transpose Operation

## Description
Transpose a 2D matrix: `C = A^T`

## Kernel Implementation Strategy
- Block size: 16x16 or 32x32
- Shared memory to avoid bank conflicts
- Handle non-square matrices

## Files
- `kernel.cu` - Optimized transpose kernel
