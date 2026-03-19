# RNN and Special Operations

Recurrent and specialized neural network operations.

## RNN Operators

### `lstm/` - Long Short-Term Memory
- Cell state and hidden state management
- Gate computations (input, forget, output, cell)
- Optional peephole connections

### `gru/` - Gated Recurrent Unit
- Simpler than LSTM with reset/update gates
- Lower memory overhead

## Utility Operators

### `cast/` - Type Casting
Convert between data types (float, int, half, etc.)

### `constant/` - Constant Tensor
Create constant-filled tensors

### `constant-of-shape/` - Constant with Shape
Create constant tensor matching another shape

### `identity/` - Identity Operation
Pass-through operation

### `dropout/` - Dropout
Regularization: randomly set elements to zero

### `if/` - Conditional Flow
Branch based on condition

### `eye-like/` - Identity Matrix
Create identity matrix matching input shape

## Quantization Operators

### `quantize-linear/` - Quantization
Scale and convert to lower precision

### `dequantize-linear/` - Dequantization
Convert from lower precision to float

## Special Operations

### `grid-sample/` - Spatial Grid Sampling
Sample 2D grid at arbitrary coordinates

### `non-max-suppression/` - NMS
Remove overlapping detections

### `topk/` - Top-K Elements
Find K largest/smallest values

## Performance Considerations

- **RNN**: Sequentially dependent, harder to parallelize
- **Quantization**: Mixed precision requires careful implementation
- **Special ops**: Highly algorithm-dependent
