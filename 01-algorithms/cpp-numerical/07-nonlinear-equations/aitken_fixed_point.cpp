#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>

struct FixedPointResult {
    double value;
    int iterations;
};

FixedPointResult aitken_accelerated_fixed_point(const std::function<double(double)>& g,
                                                double initial_guess,
                                                double tolerance,
                                                int max_iterations) {
    double x = initial_guess;
    for (int iteration = 1; iteration <= max_iterations; ++iteration) {
        const double x1 = g(x);
        const double x2 = g(x1);
        const double denominator = x2 - 2.0 * x1 + x;
        double next = x2;
        if (std::abs(denominator) > 1e-14) {
            next = x - (x1 - x) * (x1 - x) / denominator;
        }
        if (std::abs(next - x) < tolerance) {
            return {next, iteration};
        }
        x = next;
    }
    throw std::runtime_error("aitken iteration did not converge");
}

int main() {
    auto g = [](double x) { return std::cos(x); };
    const auto result = aitken_accelerated_fixed_point(g, 0.5, 1e-10, 50);
    std::cout << "root ≈ " << result.value << '\n';
    std::cout << "iterations = " << result.iterations << '\n';
    return 0;
}
