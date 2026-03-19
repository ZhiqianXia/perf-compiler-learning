# Log Operation

## Description
Element-wise natural logarithm: `C = log(A)`

## Kernel Implementation Strategy
- Unary operation
- Block size: 256 threads
- Use `logf()` intrinsic
- Handle A <= 0 cases

## Files
- `kernel.cu` - CUDA kernel implementation
