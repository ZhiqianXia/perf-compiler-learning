# Sin Operation

## Description
Element-wise sine: `C = sin(A)`

## Kernel Implementation Strategy
- Unary operation
- Block size: 256 threads
- Use `sinf()` intrinsic

## Files
- `kernel.cu` - CUDA kernel implementation
