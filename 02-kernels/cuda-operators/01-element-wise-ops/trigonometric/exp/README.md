# Exp Operation

## Description
Element-wise exponential: `C = e^A`

## Kernel Implementation Strategy
- Unary operation
- Block size: 256 threads
- Use `expf()` intrinsic

## Files
- `kernel.cu` - CUDA kernel implementation
