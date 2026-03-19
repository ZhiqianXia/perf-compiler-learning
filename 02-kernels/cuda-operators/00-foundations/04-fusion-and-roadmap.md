# Fusion and Roadmap

## Why Single Operators Are Not Enough

Real performance work often comes from reducing memory traffic between operators, not only from making each single kernel faster.

Typical fusion candidates:
- matmul + bias + gelu
- matmul + scale + softmax
- layernorm + residual
- bias + activation

## When Fusion Pays Off

Fusion is most useful when:
- intermediate tensors are large
- each operator alone is memory-bound
- the same data would otherwise be read and written multiple times

## Fusion Risks

- register pressure grows quickly
- kernel becomes shape-specific
- debugging and correctness work get harder
- a fused kernel can become slower on small shapes

## Recommended Fusion Roadmap

1. make baseline single operators correct
2. benchmark them separately
3. identify memory traffic bottlenecks
4. fuse only two or three adjacent stages first
5. compare fused and unfused versions on real shapes

## First Three Fused Blocks To Study

1. `matmul-bias-gelu`
2. `attention-score-block`
3. `layernorm-residual`

See `../09-fused-patterns/` for the suggested breakdown.
