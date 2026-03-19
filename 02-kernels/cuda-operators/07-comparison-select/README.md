# Comparison and Selection Operations

Logical operations and conditional selection.

## Operators

### `equal/` - Equality Comparison
Element-wise equality: `output = (A == B) ? 1 : 0`

### `greater/` - Greater Than
Element-wise comparison: `output = A > B`

### `greater-or-equal/` - Greater or Equal
Element-wise comparison: `output = A >= B`

### `less/` - Less Than
Element-wise comparison: `output = A < B`

### `less-or-equal/` - Less or Equal
Element-wise comparison: `output = A <= B`

### `where/` - Conditional Selection
Element-wise ternary: `output = cond ? A : B`

### `max/` - Element-wise Maximum
`output = max(A, B)` per element

### `min/` - Element-wise Minimum
`output = min(A, B)` per element

### `topk/` - Top-K Selection
Find K largest (or smallest) values

### `non-max-suppression/` - NMS
Remove overlapping bounding boxes

## Performance

- Comparisons: Memory-bound, high throughput
- Where/Select: Conditional branching overhead
- TopK/NMS: Requires sorting/scanning
