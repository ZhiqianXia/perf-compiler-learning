#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

struct CorrelationResult {
    double covariance;
    double correlation;
};

CorrelationResult correlation_analysis(const std::vector<double>& xs, const std::vector<double>& ys) {
    if (xs.size() != ys.size() || xs.empty()) {
        throw std::invalid_argument("input size mismatch");
    }
    const double n = static_cast<double>(xs.size());
    double mean_x = 0.0;
    double mean_y = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        mean_x += xs[i] / n;
        mean_y += ys[i] / n;
    }
    double covariance = 0.0;
    double variance_x = 0.0;
    double variance_y = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        const double dx = xs[i] - mean_x;
        const double dy = ys[i] - mean_y;
        covariance += dx * dy / n;
        variance_x += dx * dx / n;
        variance_y += dy * dy / n;
    }
    return {covariance, covariance / std::sqrt(variance_x * variance_y)};
}

int main() {
    const auto result = correlation_analysis({1, 2, 3, 4, 5}, {2, 4, 5, 4, 5});
    std::cout << "covariance = " << result.covariance << '\n';
    std::cout << "correlation = " << result.correlation << '\n';
    return 0;
}
