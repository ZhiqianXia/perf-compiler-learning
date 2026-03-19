#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace {

float gelu_reference(float x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    return 0.5f * x * (1.0f + std::tanh(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
}

std::vector<float> matmul_bias_gelu_reference(const std::vector<float>& a,
                                              const std::vector<float>& b,
                                              const std::vector<float>& bias,
                                              int m,
                                              int n,
                                              int k) {
    std::vector<float> output(m * n, 0.0f);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int inner = 0; inner < k; ++inner) {
                sum += a[row * k + inner] * b[inner * n + col];
            }
            output[row * n + col] = gelu_reference(sum + bias[col]);
        }
    }
    return output;
}

}  // namespace

int main() {
    constexpr int m = 4;
    constexpr int n = 6;
    constexpr int k = 5;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> a(m * k);
    std::vector<float> b(k * n);
    std::vector<float> bias(n);
    for (float& value : a) {
        value = dist(rng);
    }
    for (float& value : b) {
        value = dist(rng);
    }
    for (float& value : bias) {
        value = dist(rng);
    }

    const auto output = matmul_bias_gelu_reference(a, b, bias, m, n, k);
    std::cout << "shape=(" << m << "," << k << ") x (" << k << "," << n << ")\n";
    std::cout << "sample_output=" << output[0] << ", " << output[1] << ", "
              << output[2] << "\n";
    return 0;
}
