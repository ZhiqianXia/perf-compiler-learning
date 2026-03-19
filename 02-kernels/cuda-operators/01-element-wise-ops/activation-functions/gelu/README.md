# Gelu Operation

## Description
Gaussian Error Linear Unit: `C = A * Φ(A)` where Φ is standard normal CDF

## Kernel Implementation Strategy
- Unary operation
- Block size: 256 threads
- Approximate with: `0.5 * A * (1 + tanh(sqrt(2/π) * (A + 0.044715 * A^3)))`

## Files
- `kernel.cu` - CUDA kernel implementation
