# Cos Operation

## Description
Element-wise cosine: `C = cos(A)`

## Kernel Implementation Strategy
- Unary operation
- Block size: 256 threads
- Use `cosf()` intrinsic

## Files
- `kernel.cu` - CUDA kernel implementation
