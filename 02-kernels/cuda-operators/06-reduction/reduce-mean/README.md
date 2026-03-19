# ReduceMean Operation

## Description
Reduce tensor along axis by averaging

## Kernel Implementation Strategy
- Similar to ReduceSum but divide by count
- Count = n / output_size

## Files
- `kernel.cu` - CUDA kernel implementation
