# Equal Comparison

## Description
Element-wise equality comparison: `C = (A == B) ? 1 : 0`

## Kernel Implementation Strategy
- Unary operation with comparison
- Block size: 256 threads

## Files
- `kernel.cu` - CUDA kernel implementation
