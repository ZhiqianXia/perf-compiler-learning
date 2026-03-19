# ReduceSum Operation

## Description
Reduce tensor along axis by summing

## Kernel Implementation Strategy
- Parallel reduction with warp-level optimization
- Two-pass: per-block reduction + final reduction
- Shared memory for thread synchronization

## Files
- `kernel.cu` - Optimized reduction kernel
- `reference.cpp` - Host-side sum reference
- `benchmark.cu` - Reduction latency benchmark sample

## Validation Focus
- Tiny case: short vector with hand-computable sum
- Medium case: `1 << 20` elements for timing
- Note: the sample kernel accumulates into `output` and expects it to be zero-initialized
