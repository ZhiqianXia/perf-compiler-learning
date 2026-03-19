#include <iostream>
#include <vector>

std::vector<float> conv2d_reference(const std::vector<float>& input,
                                    const std::vector<float>& weight,
                                    const std::vector<float>& bias,
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
                                    int pad_w) {
    const int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    const int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
    std::vector<float> output(batch * out_channels * out_h * out_w, 0.0f);

    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oy = 0; oy < out_h; ++oy) {
                for (int ox = 0; ox < out_w; ++ox) {
                    float sum = bias.empty() ? 0.0f : bias[oc];
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int ky = 0; ky < kernel_h; ++ky) {
                            for (int kx = 0; kx < kernel_w; ++kx) {
                                const int in_y = oy * stride_h + ky - pad_h;
                                const int in_x = ox * stride_w + kx - pad_w;
                                if (in_y < 0 || in_y >= in_h || in_x < 0 || in_x >= in_w) {
                                    continue;
                                }
                                const int input_offset =
                                    ((b * in_channels + ic) * in_h + in_y) * in_w + in_x;
                                const int weight_offset =
                                    ((oc * in_channels + ic) * kernel_h + ky) * kernel_w + kx;
                                sum += input[input_offset] * weight[weight_offset];
                            }
                        }
                    }
                    const int output_offset =
                        ((b * out_channels + oc) * out_h + oy) * out_w + ox;
                    output[output_offset] = sum;
                }
            }
        }
    }
    return output;
}

int main() {
    constexpr int batch = 1;
    constexpr int in_channels = 1;
    constexpr int out_channels = 1;
    constexpr int in_h = 4;
    constexpr int in_w = 4;
    constexpr int kernel_h = 3;
    constexpr int kernel_w = 3;
    constexpr int stride = 1;
    constexpr int pad = 1;

    const std::vector<float> input = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    };
    const std::vector<float> weight = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1,
    };
    const std::vector<float> bias = {0.0f};

    const auto output = conv2d_reference(input,
                                         weight,
                                         bias,
                                         batch,
                                         in_channels,
                                         out_channels,
                                         in_h,
                                         in_w,
                                         kernel_h,
                                         kernel_w,
                                         stride,
                                         stride,
                                         pad,
                                         pad);

    std::cout << "output_size=" << output.size() << "\n";
    std::cout << "sample_output=" << output[0] << ", " << output[1] << ", "
              << output[2] << "\n";
    return 0;
}
