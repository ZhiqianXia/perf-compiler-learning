# perf-compiler-learning

A local study repository for four tracks:

- C++ numerical algorithms
- CUDA large-model operators
- C++ LeetCode algorithms
- LLVM compiler passes

## Repository Goals

This repository is organized around a progression from algorithm modeling to systems performance work:

1. Algorithm modeling with C++ LeetCode problems
2. Numerical computing implementations in C++
3. CUDA kernel and operator optimization
4. LLVM IR analysis and compiler pass experiments

## Structure

- `01-algorithms/`: C++ LeetCode and numerical methods
- `02-kernels/`: CUDA operator experiments and optimization notes
- `03-compiler/`: LLVM pass development and IR experiments
- `04-templates/`: reusable C++, CUDA, and LLVM templates
- `05-notes/`: cross-cutting notes on performance, numerics, and compiler topics
- `06-benchmarks/`: benchmark inputs and result snapshots
- `07-progress/`: roadmap, study log, and next actions
- `scripts/`: helper scripts for local workflows

## Suggested Workflow

1. Add one high-quality example to each main track first.
2. For every implementation, keep correctness checks and short notes together.
3. For CUDA and LLVM work, always record validation and profiling steps.
4. Promote repeated patterns into `04-templates/` instead of duplicating them.
