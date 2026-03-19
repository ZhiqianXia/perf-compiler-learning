# InstanceNormalization Operation

## Description
Normalize per instance independently

## Kernel Implementation Strategy
- Per-sample normalization
- Independent mean/variance per channel

## Files
- `kernel.cu` - CUDA kernel implementation
