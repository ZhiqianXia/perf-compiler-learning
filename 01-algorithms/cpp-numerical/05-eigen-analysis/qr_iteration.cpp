#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using Matrix = std::vector<std::vector<double>>;

static Matrix multiply(const Matrix& a, const Matrix& b) {
    Matrix c(a.size(), std::vector<double>(b[0].size(), 0.0));
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t k = 0; k < b.size(); ++k) {
            for (std::size_t j = 0; j < b[0].size(); ++j) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}

static std::pair<Matrix, Matrix> qr_decompose(const Matrix& matrix) {
    const std::size_t n = matrix.size();
    Matrix q(n, std::vector<double>(n, 0.0));
    Matrix r(n, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> v = matrix;
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < j; ++i) {
            for (std::size_t k = 0; k < n; ++k) {
                r[i][j] += q[k][i] * v[k][j];
            }
            for (std::size_t k = 0; k < n; ++k) {
                v[k][j] -= r[i][j] * q[k][i];
            }
        }
        for (std::size_t k = 0; k < n; ++k) {
            r[j][j] += v[k][j] * v[k][j];
        }
        r[j][j] = std::sqrt(r[j][j]);
        for (std::size_t k = 0; k < n; ++k) {
            q[k][j] = v[k][j] / r[j][j];
        }
    }
    return {q, r};
}

int main() {
    Matrix a{{4.0, 1.0, -2.0}, {1.0, 2.0, 0.0}, {-2.0, 0.0, 3.0}};
    for (int iter = 0; iter < 30; ++iter) {
        const auto [q, r] = qr_decompose(a);
        a = multiply(r, q);
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        std::cout << a[i][i] << '\n';
    }
    return 0;
}
