# Leaky Relu Operation

## Description
Leaky Rectified Linear Unit: `C = max(A, alpha * A)`

## Parameters
- alpha: slope for negative values (typically 0.01)

## Kernel Implementation Strategy
- Unary operation with alpha parameter
- Block size: 256 threads

## Files
- `kernel.cu` - CUDA kernel implementation
