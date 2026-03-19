# LayerNormalization Operation

## Description
Layer normalization: normalize per feature across batch

## Formula
`Y = (X - mean(X)) / sqrt(var(X) + eps) * gamma + beta`

## Kernel Implementation Strategy
- Parallel reduction for mean and variance
- Two-pass approach or fused kernel
- Shared memory for reductions

## Files
- `kernel.cu` - CUDA kernel implementation
- `reference.cpp` - Host-side normalization reference
- `benchmark.cu` - Transformer-style shape benchmark sample

## Validation Focus
- Tiny case: `batch=2, hidden=4` for manual inspection
- Medium case: `batch=256, hidden=768`
- Edge case: hidden size not aligned to warp or block size
