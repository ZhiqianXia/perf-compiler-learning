#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

Vector jacobi_iteration(const Matrix& a, const Vector& b, double tolerance, int max_iterations) {
    const std::size_t n = a.size();
    Vector x(n, 0.0), next(n, 0.0);
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        for (std::size_t i = 0; i < n; ++i) {
            double sum = b[i];
            for (std::size_t j = 0; j < n; ++j) {
                if (j != i) {
                    sum -= a[i][j] * x[j];
                }
            }
            next[i] = sum / a[i][i];
        }
        double diff = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            diff = std::max(diff, std::abs(next[i] - x[i]));
        }
        x = next;
        if (diff < tolerance) {
            return x;
        }
    }
    throw std::runtime_error("jacobi iteration did not converge");
}

int main() {
    Matrix a{{10.0, -1.0, 2.0, 0.0}, {-1.0, 11.0, -1.0, 3.0}, {2.0, -1.0, 10.0, -1.0}, {0.0, 3.0, -1.0, 8.0}};
    Vector b{6.0, 25.0, -11.0, 15.0};
    const auto x = jacobi_iteration(a, b, 1e-10, 200);
    for (double value : x) std::cout << value << '\n';
    return 0;
}
