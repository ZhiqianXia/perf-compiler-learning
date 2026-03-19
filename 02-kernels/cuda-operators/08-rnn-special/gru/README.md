# GRU Operation

## Description
Gated Recurrent Unit - simpler LSTM variant

## Architecture
- Reset gate, Update gate
- Candidate hidden state
- Lower memory than LSTM

## Kernel Implementation Strategy
- Sequential gate computations
- Matrix multiplications for transformations

## Files
- `kernel.cu` - CUDA kernel implementation
