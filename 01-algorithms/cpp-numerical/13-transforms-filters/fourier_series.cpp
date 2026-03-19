#include <cmath>
#include <iostream>
#include <numbers>
#include <utility>
#include <vector>

struct FourierCoefficients {
    std::vector<double> a;
    std::vector<double> b;
};

FourierCoefficients discrete_fourier_series(const std::vector<double>& samples, int order) {
    const int n = static_cast<int>(samples.size());
    FourierCoefficients coefficients{std::vector<double>(static_cast<std::size_t>(order + 1), 0.0),
                                     std::vector<double>(static_cast<std::size_t>(order + 1), 0.0)};
    const double scale = 2.0 / static_cast<double>(n);

    for (int k = 0; k <= order; ++k) {
        for (int j = 0; j < n; ++j) {
            const double angle = 2.0 * std::numbers::pi * k * j / static_cast<double>(n);
            coefficients.a[k] += scale * samples[j] * std::cos(angle);
            coefficients.b[k] += scale * samples[j] * std::sin(angle);
        }
    }
    coefficients.a[0] *= 0.5;
    coefficients.b[0] = 0.0;
    return coefficients;
}

int main() {
    std::vector<double> samples;
    for (int j = 0; j < 33; ++j) {
        const double x = 2.0 * std::numbers::pi * j / 33.0;
        samples.push_back(std::sin(x) + 0.5 * std::cos(2.0 * x));
    }
    const auto coefficients = discrete_fourier_series(samples, 4);
    for (std::size_t i = 0; i < coefficients.a.size(); ++i) {
        std::cout << i << ": a=" << coefficients.a[i] << ", b=" << coefficients.b[i] << '\n';
    }
    return 0;
}
