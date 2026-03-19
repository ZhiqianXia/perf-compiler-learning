# Reduction Operations

Aggregate tensor values across dimensions.

## Operators

### `reduce-sum/` - Sum Reduction
Sum all elements or along axis

### `reduce-mean/` - Mean Reduction
Average across dimensions

### `reduce-max/` - Max Reduction
Find maximum value

### `reduce-min/` - Min Reduction
Find minimum value

### `reduce-prod/` - Product Reduction
Multiply all elements

### `reduce-l2/` - L2 Norm
Compute L2 norm (sqrt of sum of squares)

### `reduce-logsum/` - Log-Sum Reduction
Log of sum: log(sum(exp(x)))

### `cumsum/` - Cumulative Sum
Running sum along axis

## Implementation Pattern

Reductions typically use:
1. **Per-block reduction**: Each block reduces its portion using shared memory
2. **Parallel reduction tree**:
   - Stride = blockDim.x / 2, reduce stride by 2 each iteration
   - Synchronize between iterations
3. **Final reduction**: Atomic operations or secondary kernel

## Performance Notes

- Latency: Dominated by reduction tree depth (log N)
- Throughput: ~50-70% of peak bandwidth
- Synchronization: Main bottleneck
- Smem efficiency: >95% typical