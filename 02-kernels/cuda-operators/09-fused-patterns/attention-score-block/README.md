# Attention Score Block

## Why This Block

This block captures the most important synchronization and memory patterns in attention:
- matmul for QK^T
- scaling
- masking if present
- softmax

## Unfused Pipeline

1. QK^T matmul
2. scale
3. mask
4. softmax

## Learning Goal

Understand that the challenge is not only math throughput, but also row-wise reduction and numerical stability.

## Fusion Direction

- fuse scale + mask + softmax first
- keep QK^T separate until the row-wise stage is well understood

## Files

- `kernel.cu`: naive `QK^T` stage followed by fused scale + mask + softmax
- `reference.cpp`: host-side reference with causal-mask example
- `benchmark.cu`: baseline attention-score timing harness

## Validation

- compare to a CPU reference on tiny sequence lengths
- stress-test long rows for softmax stability
