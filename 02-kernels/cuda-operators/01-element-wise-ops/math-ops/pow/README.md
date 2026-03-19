# Pow Operation

## Description
Element-wise power: `C = A ^ B` or `C = A ^ exponent`

## Kernel Implementation Strategy
- Binary or scalar exponent
- Block size: 256 threads
- Use `powf()` or `exp(exponent * log(base))`

## Files
- `kernel.cu` - CUDA kernel implementation
