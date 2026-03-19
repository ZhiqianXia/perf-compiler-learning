# Concat Operation

## Description
Concatenate tensors along specified axis

## Parameters
- axis: concatenation axis
- num_tensors: number of input tensors

## Kernel Implementation Strategy
- One grid block per output element
- Determine which input tensor contains element
- Copy from appropriate source

## Files
- `kernel.cu` - CUDA kernel implementation
