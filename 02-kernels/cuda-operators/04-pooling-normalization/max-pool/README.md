# MaxPool Operation

## Description
Max pooling: sliding window maximum

## Parameters
- kernel_size: pooling window size
- stride: stride of pooling
- padding: zero padding

## Kernel Implementation Strategy
- 2D sliding window
- Block: (16, 16)
- Each thread computes one output element
- Efficient shared memory usage

## Files
- `kernel.cu` - CUDA kernel implementation
