# Math Operations

Basic arithmetic operations on tensors.

## Implemented Operators

### `add/` - Addition
Element-wise addition: `C = A + B`
- Coalesced memory access
- Zero-overhead

### `sub/` - Subtraction
Element-wise subtraction: `C = A - B`

### `mul/` - Multiplication
Element-wise multiplication: `C = A * B`

### `div/` - Division
Element-wise division: `C = A / B`
- Handles division by zero

### `abs/` - Absolute Value
Element-wise absolute value: `C = |A|`

### `pow/` - Power
Element-wise exponentiation: `C = A ^ B`

## Other Math Operations (Template Provided)

`neg`, `mod`, `reciprocal`, `ceil`, `floor`, `round`, `clip`

## Performance Notes

- All use 256 threads/block for L1 cache efficiency
- Bandwidth-limited (typically 100+ GB/s achieved)
- Minimal instruction overhead
- Excellent GPU occupancy
