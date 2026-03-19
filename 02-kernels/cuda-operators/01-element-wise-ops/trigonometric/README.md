# Trigonometric and Transcendental Functions

Advanced mathematical functions commonly used in signal processing and neural networks.

## Implemented Operators

### `sin/` - Sine
Element-wise sine: `C = sin(A)`

### `cos/` - Cosine
Element-wise cosine: `C = cos(A)`

### `exp/` - Exponential
Element-wise exponential: `C = e^A`

### `log/` - Natural Logarithm
Element-wise logarithm: `C = log(A)`
- Handles non-positive values

### `sqrt/` - Square Root
Element-wise square root: `C = sqrt(A)`

## Other Trigonometric Functions (To Implement)

`tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `asinh`, `acosh`, `atanh`

## Performance Notes

- All use CUDA `libm` intrinsics (sinf, cosf, expf, logf, sqrtf)
- Latency: 8-20 cycles per instruction
- Throughput: 32-64 operations/cycle (FMA units)
- Watch for numerical edge cases (log of <= 0, etc.)
