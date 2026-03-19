#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

std::vector<float> layer_norm_reference(const std::vector<float>& x,
                                        const std::vector<float>& gamma,
                                        const std::vector<float>& beta,
                                        int batch_size,
                                        int hidden_size,
                                        float eps) {
    std::vector<float> y(x.size(), 0.0f);
    for (int batch = 0; batch < batch_size; ++batch) {
        const int offset = batch * hidden_size;
        float mean = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            mean += x[offset + i];
        }
        mean /= hidden_size;

        float variance = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            const float diff = x[offset + i] - mean;
            variance += diff * diff;
        }
        variance /= hidden_size;
        const float inv_std = 1.0f / std::sqrt(variance + eps);

        for (int i = 0; i < hidden_size; ++i) {
            y[offset + i] = (x[offset + i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }
    return y;
}

}  // namespace

int main() {
    constexpr int batch_size = 2;
    constexpr int hidden_size = 4;
    constexpr float eps = 1e-5f;

    const std::vector<float> x = {
        1.0f, 2.0f, 3.0f, 4.0f,
        -1.0f, 0.0f, 1.0f, 2.0f,
    };
    const std::vector<float> gamma(hidden_size, 1.0f);
    const std::vector<float> beta(hidden_size, 0.0f);

    const auto y = layer_norm_reference(x, gamma, beta, batch_size, hidden_size, eps);
    std::cout << "batch_size=" << batch_size << " hidden_size=" << hidden_size << "\n";
    std::cout << "sample_output=" << y[0] << ", " << y[1] << ", " << y[2] << "\n";
    return 0;
}
