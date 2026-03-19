#include <cuda_runtime.h>
#include <algorithm>

__global__ void max_pool_kernel(const float* input, float* output,
                                int batch_size, int in_h, int in_w, int channels,
                                int kernel_h, int kernel_w, int stride_h, int stride_w,
                                int pad_h, int pad_w) {
    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

    int od = blockIdx.x * blockDim.x + threadIdx.x;  // out index
    if (od >= batch_size * out_h * out_w * channels) return;

    int c = od % channels;
    int ow = (od / channels) % out_w;
    int oh = (od / (channels * out_w)) % out_h;
    int b = od / (channels * out_w * out_h);

    int ih_start = oh * stride_h - pad_h;
    int iw_start = ow * stride_w - pad_w;

    float max_val = -INFINITY;
    for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
            int ih = ih_start + kh;
            int iw = iw_start + kw;
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int input_idx = ((b * in_h + ih) * in_w + iw) * channels + c;
                max_val = max(max_val, input[input_idx]);
            }
        }
    }

    output[od] = max_val;
}

void max_pool(const float* input, float* output, int batch_size, int in_h, int in_w,
              int channels, int kernel_h, int kernel_w, int stride_h, int stride_w,
              int pad_h, int pad_w, cudaStream_t stream = 0) {
    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
    int total_out = batch_size * out_h * out_w * channels;

    int threadsPerBlock = 256;
    int blocksPerGrid = (total_out + threadsPerBlock - 1) / threadsPerBlock;
    max_pool_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        input, output, batch_size, in_h, in_w, channels,
        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
}
