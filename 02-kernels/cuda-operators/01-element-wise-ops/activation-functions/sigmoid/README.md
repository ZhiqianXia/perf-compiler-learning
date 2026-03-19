# Sigmoid Operation

## Description
Sigmoid activation: `C = 1 / (1 + exp(-A))`

## Kernel Implementation Strategy
- Unary operation
- Block size: 256 threads
- Use `sigmoidf()` or implement as `1.0 / (1.0 + expf(-x))`

## Files
- `kernel.cu` - CUDA kernel implementation
