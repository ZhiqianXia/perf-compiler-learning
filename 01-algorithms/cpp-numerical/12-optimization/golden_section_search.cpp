#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>

struct MinimumResult {
    double x;
    double fx;
    int iterations;
};

MinimumResult golden_section_search(const std::function<double(double)>& f,
                                    double left,
                                    double right,
                                    double tolerance,
                                    int max_iterations) {
    const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
    double x1 = right - (right - left) / phi;
    double x2 = left + (right - left) / phi;
    double f1 = f(x1);
    double f2 = f(x2);

    for (int iteration = 1; iteration <= max_iterations; ++iteration) {
        if (std::abs(right - left) < tolerance) {
            const double x = 0.5 * (left + right);
            return {x, f(x), iteration};
        }
        if (f1 < f2) {
            right = x2;
            x2 = x1;
            f2 = f1;
            x1 = right - (right - left) / phi;
            f1 = f(x1);
        } else {
            left = x1;
            x1 = x2;
            f1 = f2;
            x2 = left + (right - left) / phi;
            f2 = f(x2);
        }
    }
    throw std::runtime_error("golden-section search did not converge");
}

int main() {
    auto f = [](double x) { return (x - 2.0) * (x - 2.0) + 1.0; };
    const auto result = golden_section_search(f, -2.0, 5.0, 1e-10, 200);
    std::cout << result.x << " " << result.fx << " " << result.iterations << '\n';
    return 0;
}
