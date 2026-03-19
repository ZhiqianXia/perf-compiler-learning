#include <cmath>
#include <iostream>
#include <random>
#include <vector>

std::vector<float> layernorm_residual_reference(const std::vector<float>& x,
                                                const std::vector<float>& residual,
                                                const std::vector<float>& gamma,
                                                const std::vector<float>& beta,
                                                int batch_size,
                                                int hidden_size,
                                                float eps) {
    std::vector<float> output(x.size(), 0.0f);
    for (int row = 0; row < batch_size; ++row) {
        float sum = 0.0f;
        float sq_sum = 0.0f;
        for (int col = 0; col < hidden_size; ++col) {
            const float value = x[row * hidden_size + col] + residual[row * hidden_size + col];
            sum += value;
            sq_sum += value * value;
        }

        const float mean = sum / hidden_size;
        const float variance = sq_sum / hidden_size - mean * mean;
        const float inv_std = 1.0f / std::sqrt(variance + eps);
        for (int col = 0; col < hidden_size; ++col) {
            const int index = row * hidden_size + col;
            const float value = x[index] + residual[index];
            output[index] = (value - mean) * inv_std * gamma[col] + beta[col];
        }
    }
    return output;
}

int main() {
    constexpr int batch_size = 2;
    constexpr int hidden_size = 8;
    constexpr float eps = 1e-5f;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> x(batch_size * hidden_size);
    std::vector<float> residual(batch_size * hidden_size);
    std::vector<float> gamma(hidden_size, 1.0f);
    std::vector<float> beta(hidden_size, 0.0f);
    for (float& value : x) {
        value = dist(rng);
    }
    for (float& value : residual) {
        value = dist(rng);
    }

    const auto output =
        layernorm_residual_reference(x, residual, gamma, beta, batch_size, hidden_size, eps);
    std::cout << "shape=(" << batch_size << "," << hidden_size << ")\n";
    std::cout << "sample_output=" << output[0] << ", " << output[1] << ", "
              << output[2] << "\n";
    return 0;
}
