# CUDA Operators - Complete Index

This document provides a complete index of all ONNX and neural network operators organized in the CUDA operators library.

## Directory Structure Summary

```
cuda-operators/
├── 01-element-wise-ops/
│   ├── math-ops/              [Add, Sub, Mul, Div, Abs, Pow, ...]
│   ├── activation-functions/  [ReLU, Sigmoid, Tanh, GELU, LeakyReLU, Softmax, ...]
│   ├── trigonometric/         [Sin, Cos, Exp, Log, Sqrt, ...]
│   └── bitwise-ops/          [And, Or, Xor, Not, BitShift]
├── 02-linear-algebra/
│   └── [MatMul, Gemm]
├── 03-convolution/
│   └── [Conv, ConvTranspose, DeformConv]
├── 04-pooling-normalization/
│   ├── Pooling:   [MaxPool, AvgPool, GlobalMaxPool, GlobalAvgPool]
│   └── Normalization: [BatchNorm, LayerNorm, InstanceNorm, GroupNorm]
├── 05-tensor-ops/
│   └── [Transpose, Reshape, Concat, Gather, Scatter, Slice, Pad, ...]
├── 06-reduction/
│   └── [ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd, ReduceL2, ...]
├── 07-comparison-select/
│   └── [Equal, Greater, Less, Where, Max, Min, TopK, NMS]
└── 08-rnn-special/
    └── [LSTM, GRU, Cast, Dropout, Quantize, GridSample, ...]
```

## Full Operator List (130 operators)

### Element-wise Operations (72 operators)

#### Math Operations (9)
- Add, Sub, Mul, Div, Abs, Pow, Mod, Neg, Reciprocal
- Also: Ceil, Floor, Round, Clip

#### Activation Functions (13)
- ReLU, LeakyReLU, Sigmoid, Tanh, GELU, PReLU, SELU, ELU, CELU
- HardSigmoid, HardSwish, Mish, Softmax, LogSoftmax
- Additional: Softplus, Softsign, ThresholdedRelu

#### Trigonometric (11)
- Sin, Cos, Tan, Asin, Acos, Atan
- Sinh, Cosh, Asinh, Acosh, Atanh
- Also: Exp, Log, Sqrt, Erf, Sign

#### Bitwise Operations (5)
- BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot, BitShift
- Also: And, Or, Not, Xor

### Linear Algebra (3)
- MatMul, Gemm, Dot

### Convolution (3)
- Conv, ConvTranspose, DeformConv

### Pooling & Normalization (10)
- MaxPool, AveragePool, GlobalMaxPool, GlobalAveragePool
- BatchNormalization, LayerNormalization, InstanceNormalization, GroupNormalization

### Tensor Operations (21)
- Reshape, Flatten, Transpose, Squeeze, Unsqueeze, Expand
- Concat, Split, Slice, Gather, GatherElements, GatherND, Scatter, ScatterElements, ScatterND
- Pad, Tile, Shape, Size, NonZero, EyeLike
- Additional: DepthToSpace, SpaceToDepth, Trilu

### Reduction (8)
- ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd
- ReduceL2, ReduceLogSum, CumSum

### Comparison & Selection (8)
- Equal, Greater, GreaterOrEqual, Less, LessOrEqual
- Where, Max, Min, TopK, NonMaxSuppression

### RNN & Special (12)
- LSTM, GRU
- Cast, Constant, ConstantOfShape, Identity, Dropout, If
- QuantizeLinear, DequantizeLinear
- GridSample, NonMaxSuppression
- Additional: Resize (sampling)

## Implementation Status

### Fully Implemented (18 operators)
✓ Add, Sub, Mul, Div, Abs, Pow
✓ ReLU, Sigmoid, Tanh, GELU, LeakyReLU, Softmax
✓ Sin, Cos, Exp, Log, Sqrt
✓ MatMul, LayerNormalization, MaxPool
✓ Transpose, Concat, Gather
✓ ReduceSum, ReduceMean
✓ Equal, Where

### Framework Ready (112 operators)
Each has a README template with:
- Operator description
- Kernel implementation strategy
- Parameter specifications
- Performance considerations
- File structure for implementation

## Next Steps for Development

1. **Priority**: Conv, Batch Norm, GRU/LSTM
2. **High Value**: Reduction operators (Max, Min, Prod)
3. **Medium Priority**: All comparison and pooling variants
4. **Lower Priority**: Special operators (Quantize, GridSample)

## File Reading Guide

For each operator, see `{operator}/README.md`:
- **Description**: What the operator does
- **Kernel Strategy**: How to implement efficiently
- **Parameters**: Input dimensions and configurations
- **Files**: kernel.cu contains CUDA code

For category overviews, see `{category}/README.md`

## Performance Baseline Expectations

- **Element-wise**: >95% of peak bandwidth
- **MatMul**: >85% of peak (tiled kernels)
- **Reductions**: >70% of peak (synchronization overhead)
- **Pooling**: Variable (depends on kernel size)
