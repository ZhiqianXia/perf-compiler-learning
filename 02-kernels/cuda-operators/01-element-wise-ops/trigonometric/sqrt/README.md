# Sqrt Operation

## Description
Element-wise square root: `C = sqrt(A)`

## Kernel Implementation Strategy
- Unary operation
- Block size: 256 threads
- Use `sqrtf()` intrinsic

## Files
- `kernel.cu` - CUDA kernel implementation
