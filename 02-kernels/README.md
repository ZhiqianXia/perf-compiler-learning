# Kernels

This directory is for CUDA operator experiments.

## Knowledge Trunk

The CUDA direction is now organized in three layers:

1. operator taxonomy in `cuda-operators/`
2. learning foundations in `cuda-operators/00-foundations/`
3. validation and report templates in `cuda-operators/common/`

This keeps the project from becoming only a list of operator names.

## Recommended Study Path

1. read `cuda-operators/00-foundations/README.md`
2. study one operator family such as element-wise, reduction, or matmul
3. validate with the checklist and benchmark harness in `cuda-operators/common/`
4. move to fused blocks in `cuda-operators/09-fused-patterns/`

Each operator directory should include:

- a short problem statement
- a correctness reference
- one baseline implementation
- one or more optimized implementations
- benchmark notes
