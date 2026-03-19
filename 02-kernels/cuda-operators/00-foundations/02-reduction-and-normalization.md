# Reduction and Normalization Path

## Why This Pattern Matters

Most high-value neural network operators are not pure element-wise kernels. They depend on block-level collectives:
- sum
- max
- mean
- variance
- argmax/topk style selection

This makes reduction the real core pattern of the CUDA operator layer.

## Reduction Ladder

### Step 1: Shared Memory Tree Reduction

Use when:
- teaching the pattern
- block-local reduction is enough

Pros:
- easy to understand
- explicit synchronization

Cons:
- more synchronization than necessary
- not ideal for final optimized implementation

### Step 2: Warp-level Reduction

Use when:
- block size is moderate
- reduction inside a warp is frequent

Pros:
- fewer barriers
- lower shared-memory traffic

Cons:
- more advanced to read
- requires careful lane mapping

### Step 3: Multi-stage Reduction

Use when:
- reduction axis is large
- output count is small

Pattern:
- stage 1: block partials
- stage 2: final aggregation

## Normalization Family

Normalization operators are structured reductions plus affine transforms.

### Softmax
- pass 1: row max
- pass 2: sum of exp(x - max)
- pass 3: normalize

Key idea:
- subtract the max for numerical stability

### LayerNorm
- pass 1: mean
- pass 2: variance
- pass 3: normalize and apply gamma/beta

Key idea:
- reduction axis is usually the last dimension
- one block per row is a good first design

### BatchNorm / InstanceNorm / GroupNorm

They differ mainly in which axes define one statistical group.

The kernel design should start from:
- which dimensions form one reduction domain
- whether statistics are reused between training and inference
- how much temporary storage is needed

## Practical Rule

If you understand and can benchmark:
- reduce-sum
- softmax
- layernorm

then you understand most of the CUDA operator-layer synchronization problems.
