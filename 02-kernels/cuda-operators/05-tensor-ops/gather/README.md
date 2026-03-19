# Gather Operation

## Description
Gather elements from input using indices

## Kernel Implementation Strategy
- Index lookup operation
- One thread per output element
- Random memory access (scattered reads)

## Files
- `kernel.cu` - CUDA kernel implementation
