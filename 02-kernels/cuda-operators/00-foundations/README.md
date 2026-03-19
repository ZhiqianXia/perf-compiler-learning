# CUDA Foundations

This section turns the operator collection into a learnable CUDA kernel track.

## Why This Exists

The operator directories are good for browsing by function, but real CUDA learning needs a second view:

1. launch and memory basics
2. reduction and row-wise collective patterns
3. correctness and benchmarking discipline
4. fusion patterns and end-to-end blocks

## Recommended Learning Order

1. `01-memory-and-launch.md`
2. `02-reduction-and-normalization.md`
3. `03-validation-and-benchmarking.md`
4. `04-fusion-and-roadmap.md`
5. apply the checklist to one operator family at a time

## Outcome

After finishing this section, each operator directory should be readable through four questions:
- What is the mathematical contract?
- What memory pattern dominates?
- What optimization lever matters most?
- How do we prove correctness and measure performance?
