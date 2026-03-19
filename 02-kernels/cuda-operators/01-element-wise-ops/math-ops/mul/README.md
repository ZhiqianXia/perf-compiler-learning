# Mul Operation

## Description
Element-wise multiplication: `C = A * B`

## Kernel Implementation Strategy
- Block size: 256 threads
- Grid: (total_elements + 255) / 256

## Files
- `kernel.cu` - CUDA kernel implementation
