# Abs Operation

## Description
Element-wise absolute value: `C = |A|`

## Kernel Implementation Strategy
- Unary operation
- Block size: 256 threads
- Grid: (total_elements + 255) / 256

## Files
- `kernel.cu` - CUDA kernel implementation
