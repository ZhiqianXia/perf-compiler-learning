# Convolution Operations

Spatial convolution operators for image and feature processing.

## Operators

### `conv/` - 2D Convolution
Standard 2D convolution with configurable:
- Kernel size, stride, padding
- Dilation support
- Input channels, output channels

### `conv-transpose/` - Transposed Convolution
Deconvolution operation for upsampling

### `deform-conv/` - Deformable Convolution
Convolution with learnable offsets for spatial flexibility

## Implementation Notes

- 2D thread blocks for spatial locality
- Shared memory for input patches
- Support for different data layouts (NCHW, NHWC)