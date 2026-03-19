# Conv Operation

## Description
2D Convolution for spatial features

## Parameters
- Input: (B, C_in, H, W)
- Kernel: (C_out, C_in, KH, KW)
- Output: (B, C_out, H', W')

## Kernel Implementation Strategy
- Block per output position
- Shared memory for input patches
- Optimized for different batch sizes

## Files
- `kernel.cu` - CUDA kernel implementation
- `reference.cpp` - Host-side NCHW direct convolution reference
- `benchmark.cu` - Baseline 3x3 NCHW convolution benchmark

## Validation Focus
- Tiny case: `1x1x4x4` input with `3x3` filter for manual checking
- Medium case: `N=16, C=32, H=W=56, K=64, R=S=3`
- Edge case: stride and padding combinations that change output shape
