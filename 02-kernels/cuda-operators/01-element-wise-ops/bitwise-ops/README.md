# Bitwise Operations

Low-level bit manipulation operations.

## Operators

### `bitwiseand/` - Bitwise AND
Bitwise AND: `C = A & B`

### `bitwiseor/` - Bitwise OR
Bitwise OR: `C = A | B`

### `bitwisexor/` - Bitwise XOR
Bitwise XOR: `C = A ^ B`

### `bitwisenot/` - Bitwise NOT
Bitwise NOT: `C = ~A`

### `bitshift/` - Bit Shifting
Left/right bit shifts with direction indicator

## Use Cases

- Quantization/dequantization bit manipulation
- Efficient packing/unpacking of data
- Cryptographic operations
- Feature extraction from packed representations

## Performance

- All very fast (1-2 cycle latency)
- Fully parallel across lanes
- Zero-overhead operations
