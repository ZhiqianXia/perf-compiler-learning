#include <cuda_runtime.h>

__global__ void conv2d_nchw_kernel(const float* input,
                                   const float* weight,
                                   const float* bias,
                                   float* output,
                                   int batch,
                                   int in_channels,
                                   int out_channels,
                                   int in_h,
                                   int in_w,
                                   int kernel_h,
                                   int kernel_w,
                                   int stride_h,
                                   int stride_w,
                                   int pad_h,
                                   int pad_w,
                                   int out_h,
                                   int out_w) {
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int linear_index = blockIdx.z;

    if (out_x >= out_w || out_y >= out_h) {
        return;
    }

    const int batch_index = linear_index / out_channels;
    const int out_channel = linear_index % out_channels;
    if (batch_index >= batch) {
        return;
    }

    float sum = bias != nullptr ? bias[out_channel] : 0.0f;
    for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int in_y = out_y * stride_h + kh - pad_h;
                const int in_x = out_x * stride_w + kw - pad_w;
                if (in_y < 0 || in_y >= in_h || in_x < 0 || in_x >= in_w) {
                    continue;
                }

                const int input_offset =
                    ((batch_index * in_channels + in_channel) * in_h + in_y) * in_w + in_x;
                const int weight_offset =
                    ((out_channel * in_channels + in_channel) * kernel_h + kh) * kernel_w + kw;
                sum += input[input_offset] * weight[weight_offset];
            }
        }
    }

    const int output_offset =
        ((batch_index * out_channels + out_channel) * out_h + out_y) * out_w + out_x;
    output[output_offset] = sum;
}

void conv2d_nchw(const float* input,
                 const float* weight,
                 const float* bias,
                 float* output,
                 int batch,
                 int in_channels,
                 int out_channels,
                 int in_h,
                 int in_w,
                 int kernel_h,
                 int kernel_w,
                 int stride_h,
                 int stride_w,
                 int pad_h,
                 int pad_w,
                 cudaStream_t stream = 0) {
    const int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    const int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

    dim3 block_dim(16, 16);
    dim3 grid_dim((out_w + block_dim.x - 1) / block_dim.x,
                  (out_h + block_dim.y - 1) / block_dim.y,
                  batch * out_channels);
    conv2d_nchw_kernel<<<grid_dim, block_dim, 0, stream>>>(
        input,
        weight,
        bias,
        output,
        batch,
        in_channels,
        out_channels,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        out_h,
        out_w);
}
