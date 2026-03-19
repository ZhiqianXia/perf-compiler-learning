# Add Operation

## Description
Element-wise addition of two tensors: `C = A + B`

## Kernel Implementation Strategy
- Block size: 256 threads
- Grid: (total_elements + 255) / 256
- Memory: Coalesced reads from both inputs

## Files
- `kernel.cu` - CUDA kernel implementation
- `test.cu` - Correctness tests
