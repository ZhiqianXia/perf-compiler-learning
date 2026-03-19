# Reshape Operation

## Description
Reshape tensor to new shape (no data copy, just logical reshape)

## Kernel Implementation Strategy
- Usually just logical indexing
- GPU kernel only needed for complex strides

## Files
- `kernel.cu` - Index mapping functions
