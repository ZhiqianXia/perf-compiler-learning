#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace {

std::vector<float> matmul_reference(const std::vector<float>& a,
                                    const std::vector<float>& b,
                                    int m,
                                    int n,
                                    int k) {
    std::vector<float> c(m * n, 0.0f);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int inner = 0; inner < k; ++inner) {
                sum += a[row * k + inner] * b[inner * n + col];
            }
            c[row * n + col] = sum;
        }
    }
    return c;
}

float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    float diff = 0.0f;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        diff = std::max(diff, std::abs(lhs[i] - rhs[i]));
    }
    return diff;
}

}  // namespace

int main() {
    constexpr int m = 4;
    constexpr int n = 5;
    constexpr int k = 3;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> a(m * k);
    std::vector<float> b(k * n);
    for (float& value : a) {
        value = dist(rng);
    }
    for (float& value : b) {
        value = dist(rng);
    }

    const auto reference = matmul_reference(a, b, m, n, k);
    const auto replay = matmul_reference(a, b, m, n, k);

    std::cout << "shape=(" << m << "," << k << ") x (" << k << "," << n << ")\n";
    std::cout << "max_abs_diff=" << max_abs_diff(reference, replay) << "\n";
    std::cout << "sample_output=" << reference[0] << ", " << reference[1] << ", "
              << reference[2] << "\n";
    return 0;
}
