# CUDA Operators Library

A comprehensive collection of CUDA kernel implementations for neural network and tensor operations, organized by functionality.

## How To Read This Directory

There are now three complementary views of the CUDA operator layer:

1. `00-foundations/`: the learning path for CUDA memory, reduction, benchmarking, and fusion
2. `01-08`: operator taxonomy by function family
3. `09-fused-patterns/`: small end-to-end blocks closer to real model execution

If you only read the operator folders, you will see what exists.
If you also read `00-foundations/`, you will understand why kernels are structured the way they are.

## Start Here

- `00-foundations/01-memory-and-launch.md`
- `00-foundations/02-reduction-and-normalization.md`
- `00-foundations/03-validation-and-benchmarking.md`
- `00-foundations/04-fusion-and-roadmap.md`
- `common/correctness_checklist.md`
- `common/operator-report-template.md`

## Directory Structure

### 0. Foundations (`00-foundations/`)

- memory and launch design
- reduction and normalization patterns
- correctness and benchmarking discipline
- fusion roadmap

### 1. Element-wise Operations (`01-element-wise-ops/`)

#### Math Operations (`math-ops/`)
Implemented: `add`, `sub`, `mul`, `div`, `abs`, `pow`

Other operators: `mod`, `neg`, `reciprocal`, `ceil`, `floor`, `round`, `clip`

#### Activation Functions (`activation-functions/`)
Implemented: `relu`, `sigmoid`, `tanh`, `gelu`, `leaky-relu`, `softmax`

Other operators: `prelu`, `selu`, `elu`, `celu`, `hardswish`, `hardsigmoid`, `mish`, `erf`, `softsign`, `softplus`

#### Trigonometric & Math (`trigonometric/`)
Implemented: `sin`, `cos`, `exp`, `log`, `sqrt`

Other operators: `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `asinh`, `acosh`, `atanh`

#### Bitwise Operations (`bitwise-ops/`)
`bitwiseand`, `bitwiseor`, `bitwisexor`, `bitwisenot`, `bitshift`

### 2. Linear Algebra (`02-linear-algebra/`)
- `matmul` - Optimized with tiling
- `gemm`, `dot`

### 3. Convolution (`03-convolution/`)
- `conv`, `conv-transpose`, `deform-conv`

### 4. Pooling & Normalization (`04-pooling-normalization/`)

**Pooling**: `max-pool`, `avg-pool`, `global-max-pool`, `global-avg-pool`

**Normalization**: `layer-norm`, `batch-norm`, `instance-norm`, `group-norm`

### 5. Tensor Operations (`05-tensor-ops/`)
Implemented: `transpose`, `concat`, `gather`

Others: `reshape`, `split`, `squeeze`, `unsqueeze`, `slice`, `pad`, `expand`, `flatten`, `tile`, `scatter`, `scan`, `eye-like`, `depth-to-space`, `space-to-depth`

### 6. Reduction (`06-reduction/`)
Implemented: `reduce-sum`, `reduce-mean`

Others: `reduce-max`, `reduce-min`, `reduce-prod`, `reduce-l2`, `reduce-logsum`, `cumsum`

### 7. Comparison & Selection (`07-comparison-select/`)
Implemented: `equal`, `where`

Others: `greater`, `less`, `greater-or-equal`, `less-or-equal`, `max`, `min`, `topk`, `non-max-suppression`

### 8. RNN & Special (`08-rnn-special/`)
- `lstm`, `gru` - Recurrent operations
- `cast`, `constant`, `identity`, `dropout` - Utility operations
- `quantize-linear`, `dequantize-linear` - Quantization
- `grid-sample`, `non-max-suppression` - Special operations

### 9. Fused Patterns (`09-fused-patterns/`)
- `matmul-bias-gelu`
- `attention-score-block`
- `layernorm-residual`

## Recommended Layout per Operator

Each operator directory contains:
- `README.md` - Description, algorithm, optimization path
- `kernel.cu` - CUDA kernel implementation(s)
- `test.cu` (optional) - Correctness tests

Use the shared templates in `common/` when adding new operators.

## Validation and Benchmarking

The operator layer is not considered complete without:
- a correctness reference
- shape coverage
- tolerance definition
- measured timing
- baseline versus optimized comparison

See:
- `common/correctness_checklist.md`
- `common/operator-report-template.md`
- `common/benchmark-harness.md`

## Basic Kernel Patterns

### Element-wise Kernel
```cuda
__global__ void op_kernel(const float* a, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = operation(a[idx]);
}
```

### Launch Helper
```cuda
void op(const float* a, float* c, int n, cudaStream_t stream = 0) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    op_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a, c, n);
}
```

## Performance Notes

- Element-wise: 256 threads/block for L1 efficiency
- Reductions: Use shared memory + warp primitives
- 2D Ops: 16x16 or 32x32 thread blocks
- Memory: Ensure coalesced access patterns

## Legacy Note

The root-level directories `vector-add`, `matmul`, and `layernorm` are older standalone experiments.
The numbered directories are the current primary organization.
