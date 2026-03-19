# BatchNormalization Operation

## Description
Batch normalization across samples

## Formula
`Y = (X - mean(X)) / sqrt(var(X) + eps) * gamma + beta`

## Kernel Implementation Strategy
- Parallel reduction for statistics
- Two-pass: compute moments, then normalize

## Files
- `kernel.cu` - CUDA kernel implementation
