#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

Vector sor_iteration(const Matrix& a, const Vector& b, double omega, double tolerance, int max_iterations) {
    const std::size_t n = a.size();
    Vector x(n, 0.0);
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        double diff = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double sigma = b[i];
            for (std::size_t j = 0; j < n; ++j) {
                if (j != i) {
                    sigma -= a[i][j] * x[j];
                }
            }
            const double updated = (1.0 - omega) * x[i] + omega * sigma / a[i][i];
            diff = std::max(diff, std::abs(updated - x[i]));
            x[i] = updated;
        }
        if (diff < tolerance) {
            return x;
        }
    }
    throw std::runtime_error("SOR iteration did not converge");
}

int main() {
    Matrix a{{4.0, 1.0, 2.0}, {3.0, 5.0, 1.0}, {1.0, 1.0, 3.0}};
    Vector b{4.0, 7.0, 3.0};
    const auto x = sor_iteration(a, b, 1.1, 1e-10, 200);
    for (double value : x) std::cout << value << '\n';
    return 0;
}
