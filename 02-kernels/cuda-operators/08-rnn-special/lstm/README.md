# LSTM Operation

## Description
Long Short-Term Memory cell for sequence processing

## Architecture
- Input gate, Forget gate, Cell gate, Output gate
- Cell state and hidden state updates
- Optional: Peephole connections, layer norm

## Kernel Implementation Strategy
- Sequence-dependent computations
- Multiple matrix multiplications per step
- Careful numerical precision handling

## Files
- `kernel.cu` - CUDA kernel implementation
