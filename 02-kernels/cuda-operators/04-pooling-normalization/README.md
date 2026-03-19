# Pooling and Normalization

Spatial aggregation and feature normalization operations.

## Pooling Operators

### `max-pool/` - Maximum Pooling
Select maximum in sliding window

### `avg-pool/` - Average Pooling
Compute average in sliding window

### `global-max-pool/` - Global Max Pooling
Maximum over entire spatial dimensions

### `global-avg-pool/` - Global Average Pooling
Average over entire spatial dimensions

## Normalization Operators

### `batch-norm/` - Batch Normalization
Normalize per feature across batch dimension

### `layer-norm/` - Layer Normalization
Normalize across feature dimension per sample

### `instance-norm/` - Instance Normalization
Normalize per instance per channel

### `group-norm/` - Group Normalization
Normalize within groups of channels

## Performance Considerations

**Pooling**:
- Scattered reads (random indices)
- Use local shared memory cache for overlapping windows

**Normalization**:
- First pass: compute mean and variance via reduction
- Second pass: normalize and apply scale/bias
- Requires careful synchronization
