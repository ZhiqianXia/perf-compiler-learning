# Relu Operation

## Description
Rectified Linear Unit: `C = max(A, 0)`

## Kernel Implementation Strategy
- Unary operation
- Block size: 256 threads
- Simple max operation with zero

## Files
- `kernel.cu` - CUDA kernel implementation
