#include <iostream>
#include <stdexcept>
#include <vector>

std::vector<double> solve_tridiagonal(std::vector<double> lower,
                                      std::vector<double> diagonal,
                                      std::vector<double> upper,
                                      std::vector<double> rhs) {
    const std::size_t n = diagonal.size();
    for (std::size_t i = 1; i < n; ++i) {
        const double factor = lower[i - 1] / diagonal[i - 1];
        diagonal[i] -= factor * upper[i - 1];
        rhs[i] -= factor * rhs[i - 1];
    }
    std::vector<double> x(n, 0.0);
    x[n - 1] = rhs[n - 1] / diagonal[n - 1];
    for (int i = static_cast<int>(n) - 2; i >= 0; --i) {
        x[i] = (rhs[i] - upper[i] * x[i + 1]) / diagonal[i];
    }
    return x;
}

int main() {
    const auto x = solve_tridiagonal({-1.0, -1.0}, {2.0, 2.0, 2.0}, {-1.0, -1.0}, {1.0, 0.0, 1.0});
    for (double value : x) std::cout << value << '\n';
    return 0;
}
