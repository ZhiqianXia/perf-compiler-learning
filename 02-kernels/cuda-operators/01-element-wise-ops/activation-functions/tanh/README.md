# Tanh Operation

## Description
Hyperbolic tangent: `C = tanh(A) = (exp(2A) - 1) / (exp(2A) + 1)`

## Kernel Implementation Strategy
- Unary operation
- Block size: 256 threads
- Use `tanhf()` or numerically stable implementation

## Files
- `kernel.cu` - CUDA kernel implementation
