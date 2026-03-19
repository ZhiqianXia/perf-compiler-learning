# ReduceLogSum Operation

## Description
Compute log(sum(exp(x))) stably

## Kernel Implementation Strategy
- Numerically stable log-sum-exp
- Subtract max for stability

## Files
- `kernel.cu` - CUDA kernel implementation
