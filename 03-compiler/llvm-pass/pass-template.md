# Pass Name

## Goal

-

## Pass Type

- analysis or transform

## LLVM APIs Used

-

## High-Level Logic

1.
2.
3.

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Run

```bash
opt -load-pass-plugin ./build/libYourPass.so -passes=your-pass input.ll -disable-output
```

## Validation

- expected stdout or IR diff:
- verifier check:
- sample test input:

## Pitfalls

-
