# Validation and Benchmarking Path

## Missing Piece In Most Kernel Repositories

A kernel is not complete when it compiles.
A kernel is complete only when you can answer:
- Is it correct?
- Is it stable?
- Is it faster than the baseline?
- Under which shape regime does it win?

## Correctness Standard

Every operator should define:
- input shape set
- dtype set
- edge cases
- reference implementation
- tolerance

### Reference Priority

1. simple CPU implementation
2. trusted framework implementation
3. mathematically equivalent decomposition

### Tolerance Guidelines

- fp32 element-wise: `1e-6` to `1e-5`
- fp32 reduction/normalization: `1e-5` to `1e-4`
- fp16/bf16: looser and shape-dependent

## Benchmark Standard

Report at least:
- problem shape
- dtype
- device model
- elapsed time
- effective bandwidth or effective FLOPs
- baseline comparison

## Benchmark Workflow

1. warm up the kernel
2. run multiple timed iterations
3. synchronize before reading the timer
4. report average and best or median
5. keep input generation fixed and documented

## What To Compare Against

- naive baseline kernel
- optimized kernel variant
- framework kernel when available

## Decision Rule

Do not keep an optimized variant if it is:
- harder to understand
- not measurably faster on target shapes
- less numerically stable

## Templates

See:
- `../common/benchmark-harness.md`
- `../common/correctness_checklist.md`
- `../common/operator-report-template.md`
