# ReduceMax Operation

## Description
Reduce along axis by taking maximum

## Kernel Implementation Strategy
- Parallel max reduction in shared memory
- Two-pass for multiple axes

## Files
- `kernel.cu` - CUDA kernel implementation
