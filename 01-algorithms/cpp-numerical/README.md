# C++ Numerical

This directory reorganizes the legacy code in `/back/XuAlgrithms-master/CPP/V5` into a modern C++ study layout.

## Organization Rule

The primary structure follows the book-style chapter order shown in the reference images:

1. `01-polynomials`
2. `02-complex-arithmetic`
3. `03-random-numbers`
4. `04-matrix-operations`
5. `05-eigen-analysis`
6. `06-linear-systems`
7. `07-nonlinear-equations`
8. `08-interpolation-approximation`
9. `09-numerical-integration`
10. `10-ordinary-differential-equations`
11. `11-data-processing`
12. `12-optimization`
13. `13-transforms-filters`
14. `14-special-functions`

## What Was Changed From V5

- old K&R C interfaces were rewritten as modern C++ examples
- raw arrays were replaced with `std::vector` or `std::complex`
- return codes were replaced with value semantics or exceptions
- each chapter adds mathematical background and numerical stability notes

## Scope

This directory focuses on the numerical chapters corresponding to V5 `ch1` through `ch14`.

- `ch15` sorting
- `ch16` searching

These two chapters are not moved here because they are general algorithms rather than numerical computing topics.

## Reading Order

1. Read [V5-analysis.md](./V5-analysis.md) for the old-to-new chapter mapping.
2. Open each chapter `README.md` for background knowledge.
3. Compile the representative `*.cpp` example for a modern reference implementation.

## Compile Example

```bash
g++ -std=c++20 -O2 04-matrix-operations/cholesky.cpp -o cholesky
./cholesky
```

## Build All Examples

```bash
cmake -S . -B build
cmake --build build -j
```

The root [CMakeLists.txt](./CMakeLists.txt) builds all representative examples from Chapter 1 to Chapter 14.

## Content Standard

For each chapter, keep:

- mathematical goal
- algorithm steps
- numerical stability notes
- complexity
- C++ implementation
- validation approach
