# Activation Functions

Neural network activation functions for introducing non-linearity.

## Implemented Operators

### `relu/` - ReLU
Rectified Linear Unit: `C = max(A, 0)`
- Simple, efficient
- No numerical issues

### `sigmoid/` - Sigmoid
`C = 1 / (1 + exp(-A))`
- Numerically stable implementation
- Avoids overflow/underflow

### `tanh/` - Tanh
Hyperbolic tangent: `C = tanh(A)`
- Symmetric around zero
- High-quality `tanhf()` intrinsic

### `gelu/` - GELU
Gaussian Error Linear Unit
- Approximate: `0.5 * A * (1 + tanh(sqrt(2/π) * (A + 0.044715*A³)))`
- Used in modern transformers

### `leaky-relu/` - Leaky ReLU
`C = max(A, alpha*A)`
- Alpha typically 0.01
- Allows small gradients for negative values

### `softmax/` - Softmax
Categorical probability distribution
- Numerically stable (max subtraction)
- Parallel reduction in shared memory
- Per-sample normalization

## Other Activations (To Implement)

`prelu`, `selu`, `elu`, `celu`, `hardswish`, `hardsigmoid`, `mish`, `erf`, `softsign`, `softplus`, `log-softmax`

## Optimization Strategies

- **Simple activations** (ReLU): 1-2 ops per element
- **Transcendental** (Sigmoid, Tanh): Use fast intrinsics
- **Compound** (GELU): Pre-compute constants
- **Softmax**: Two-pass reduction for numerical stability
