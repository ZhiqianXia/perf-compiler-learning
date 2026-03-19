#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

struct LinearRegressionResult {
    double intercept;
    double slope;
    double residual_sum_squares;
    double root_mean_square_error;
};

LinearRegressionResult fit_line(const std::vector<double>& xs, const std::vector<double>& ys) {
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

    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        numerator += (xs[i] - mean_x) * (ys[i] - mean_y);
        denominator += (xs[i] - mean_x) * (xs[i] - mean_x);
    }

    const double slope = numerator / denominator;
    const double intercept = mean_y - slope * mean_x;
    double residual_sum_squares = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        const double prediction = intercept + slope * xs[i];
        residual_sum_squares += (ys[i] - prediction) * (ys[i] - prediction);
    }
    return {intercept, slope, residual_sum_squares, std::sqrt(residual_sum_squares / n)};
}

int main() {
    std::vector<double> xs{1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> ys{1.2, 1.9, 3.2, 3.9, 5.1};
    const auto result = fit_line(xs, ys);
    std::cout << "intercept = " << result.intercept << '\n';
    std::cout << "slope = " << result.slope << '\n';
    std::cout << "rmse = " << result.root_mean_square_error << '\n';
    return 0;
}
