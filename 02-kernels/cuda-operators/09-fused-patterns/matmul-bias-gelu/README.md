# MatMul + Bias + GELU

## Why This Block

This is one of the most common MLP-side blocks in transformer models.

## Unfused Pipeline

1. matmul
2. bias add
3. gelu

## Why Fusion Helps

The bias add and GELU stages are usually memory-bound.
Fusing them after matmul can reduce global-memory reads and writes of the intermediate tensor.

## First Version To Build

- keep matmul separate
- fuse bias + gelu as a second-stage kernel

Implemented in this directory as:
- `kernel.cu`: tiled matmul into workspace, then fused bias+gelu epilogue
- `reference.cpp`: host-side unfused reference for correctness comparison
- `benchmark.cu`: baseline timing harness for MLP-style shapes

## Second Version To Build

- fuse epilogue into the tiled matmul kernel

## Validation

- compare against unfused matmul then element-wise bias and gelu
- test hidden sizes that are not multiples of the tile size

## Next Upgrade

- move bias + gelu into the matmul write-back path to remove the workspace write/read pair
