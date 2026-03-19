# LayerNorm + Residual

## Why This Block

Residual connections and normalization often sit next to each other in transformer blocks.

## Unfused Pipeline

1. residual add
2. layernorm

## Why Fusion Helps

Both stages touch the same row-wise tensor data. Fusion can cut one full round-trip to global memory.

## Learning Goal

Study how reduction kernels interact with simple element-wise updates when registers and shared memory are limited.

## Files

- `kernel.cu`: single-kernel fused residual add + layernorm sample
- `reference.cpp`: host-side unfused reference
- `benchmark.cu`: transformer-style timing harness

## Validation

- compare to separate residual add and layernorm kernels
- test hidden sizes from 256 to 8192
- record both error and bandwidth
