# Where Operation

## Description
Conditional selection: `C = condition ? A : B`

## Kernel Implementation Strategy
- Ternary operation per element
- Block size: 256 threads

## Files
- `kernel.cu` - CUDA kernel implementation
