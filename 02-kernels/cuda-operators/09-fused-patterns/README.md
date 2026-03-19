# Fused Patterns

This section describes small operator blocks that are more representative of real model execution than isolated single kernels.

## Why This Matters

Single operators teach CUDA mechanics.
Fused patterns teach performance engineering.

## Recommended Order

1. `matmul-bias-gelu`
2. `layernorm-residual`
3. `attention-score-block`

## Current Status

- `matmul-bias-gelu`: sample kernel, host reference, benchmark scaffold are in place
- `layernorm-residual`: sample kernel, host reference, benchmark scaffold are in place
- `attention-score-block`: sample kernel, host reference, benchmark scaffold are in place

## Evaluation Rule

For each fused block, compare:
- unfused pipeline total time
- fused kernel time
- numerical deviation
- shape regimes where fusion wins
