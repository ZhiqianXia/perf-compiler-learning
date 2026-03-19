#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

static Vector solve_shifted_system(Matrix a, Vector b, double shift) {
    const std::size_t n = a.size();
    for (std::size_t i = 0; i < n; ++i) {
        a[i][i] -= shift;
    }
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivot = k;
        for (std::size_t i = k + 1; i < n; ++i) {
            if (std::abs(a[i][k]) > std::abs(a[pivot][k])) {
                pivot = i;
            }
        }
        std::swap(a[k], a[pivot]);
        std::swap(b[k], b[pivot]);
        for (std::size_t i = k + 1; i < n; ++i) {
            const double factor = a[i][k] / a[k][k];
            for (std::size_t j = k; j < n; ++j) {
                a[i][j] -= factor * a[k][j];
            }
            b[i] -= factor * b[k];
        }
    }
    Vector x(n, 0.0);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        double sum = b[i];
        for (std::size_t j = i + 1; j < n; ++j) {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }
    return x;
}

int main() {
    Matrix a{{2.0, 1.0}, {1.0, 2.0}};
    Vector x{1.0, 0.0};
    const double shift = 1.1;
    for (int iter = 0; iter < 8; ++iter) {
        x = solve_shifted_system(a, x, shift);
        double norm = 0.0;
        for (double value : x) {
            norm += value * value;
        }
        norm = std::sqrt(norm);
        for (double& value : x) {
            value /= norm;
        }
    }
    std::cout << x[0] << '\n' << x[1] << '\n';
    return 0;
}
