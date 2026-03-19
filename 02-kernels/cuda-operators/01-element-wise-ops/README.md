# Element-wise Operations

Element-wise operations apply the same function to each element independently.

## Subdirectories

### `math-ops/` - Basic Mathematical Operations
Core arithmetic operations: Add, Sub, Mul, Div, Abs, Pow, Reciprocal, Neg, Mod

**Optimization Focus**: Memory bandwidth (coalesced access)

### `activation-functions/` - Activation Functions
Neural network activation functions: ReLU, Sigmoid, Tanh, GELU, Softmax, LeakyReLU, etc.

**Optimization Focus**: Instruction-level parallelism, numerically stable implementations

### `trigonometric/` - Trigonometric & Transcendental Functions
Math functions: Sin, Cos, Tan, Exp, Log, Sqrt, ArcSin, ArcCos, ArcTan, etc.

**Optimization Focus**: CUDA math intrinsics efficiency

### `bitwise-ops/` - Bitwise Operations
Logical operations on bits: AND, OR, XOR, NOT, BitShift

## Common Pattern

All element-wise operations follow this launcher pattern:

```cuda
void op(const float* input, float* output, int n, cudaStream_t stream = 0) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    op_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(input, output, n);
}
```

## Performance Characteristics

- **Latency**: ~1-2 μs per 1M elements
- **Throughput**: >90% of peak bandwidth for coalesced access
- **Occupancy**: Typically high (12+ warps per multiprocessor)
