#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

static Vector solve_normal_equations(Matrix a, Vector b) {
    const std::size_t n = a.size();
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivot = k;
        for (std::size_t i = k + 1; i < n; ++i) {
            if (std::abs(a[i][k]) > std::abs(a[pivot][k])) pivot = i;
        }
        std::swap(a[k], a[pivot]);
        std::swap(b[k], b[pivot]);
        for (std::size_t i = k + 1; i < n; ++i) {
            const double factor = a[i][k] / a[k][k];
            for (std::size_t j = k; j < n; ++j) a[i][j] -= factor * a[k][j];
            b[i] -= factor * b[k];
        }
    }
    Vector x(n, 0.0);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        double sum = b[i];
        for (std::size_t j = i + 1; j < n; ++j) sum -= a[i][j] * x[j];
        x[i] = sum / a[i][i];
    }
    return x;
}

Vector multiple_linear_regression(const Matrix& features, const Vector& targets) {
    const std::size_t m = features.size();
    const std::size_t n = features.front().size() + 1;
    Matrix xtx(n, Vector(n, 0.0));
    Vector xty(n, 0.0);
    for (std::size_t row = 0; row < m; ++row) {
        Vector augmented(n, 1.0);
        for (std::size_t col = 1; col < n; ++col) augmented[col] = features[row][col - 1];
        for (std::size_t i = 0; i < n; ++i) {
            xty[i] += augmented[i] * targets[row];
            for (std::size_t j = 0; j < n; ++j) xtx[i][j] += augmented[i] * augmented[j];
        }
    }
    return solve_normal_equations(xtx, xty);
}

int main() {
    Matrix x{{1.0, 2.0}, {2.0, 0.0}, {3.0, 1.0}, {4.0, 3.0}};
    Vector y{4.0, 5.0, 7.0, 10.0};
    const auto beta = multiple_linear_regression(x, y);
    for (double value : beta) std::cout << value << '\n';
    return 0;
}
