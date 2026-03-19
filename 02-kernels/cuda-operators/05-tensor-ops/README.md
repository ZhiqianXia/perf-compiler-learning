# Tensor Operations

Shape and structure manipulation without data copying.

## Operators

### `transpose/` - Matrix Transpose
Rearrange matrix dimensions with bank conflict avoidance

### `reshape/` - Reshape
Logical reshape (pointer-only, no data movement)

### `concat/` - Concatenation
Join multiple tensors along axis

### `gather/` - Gather by Indices
Select elements using index arrays

### `scatter/` - Scatter by Indices
Place elements at index-specified locations

### `slice/` - Slicing
Extract subregion from tensor

### `squeeze/` - Squeeze
Remove dimensions of size 1

### `unsqueeze/` - Unsqueeze
Insert dimensions of size 1

### `split/` - Split
Divide tensor into parts

### `pad/` - Padding
Add padding around tensor boundaries

### `flatten/` - Flatten
Convert to 1D

### `expand/` - Expand
Broadcast to larger shape

### `tile/` - Tile
Repeat tensor multiple times

## Performance Notes

- Most operations are bandwidth-limited (copy-like)
- Transpose: use shared memory + careful indexing
- Gather/Scatter: irregular memory access patterns
- Others: simple data movement
