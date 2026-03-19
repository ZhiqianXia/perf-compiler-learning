# Softmax Operation

## Description
Softmax normalization per sample

## Kernel Implementation Strategy
- Requires reduction for max and sum
- Two-pass kernel: 1) Find max per row, 2) Compute softmax
- Or optimize with shared memory

## Files
- `kernel.cu` - CUDA kernel implementation
- `reference.cpp` - Numerically stable CPU reference
- `benchmark.cu` - Row-wise latency benchmark sample

## Validation Focus
- Tiny case: short vector with known probabilities
- Stress case: large magnitude inputs to verify max-subtraction stability
- Limitation: current sample kernel demonstrates a single-row launch pattern
