#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using Matrix = std::vector<std::vector<double>>;

struct QRDecomposition {
    Matrix q;
    Matrix r;
};

static double dot(const std::vector<double>& lhs, const std::vector<double>& rhs) {
    double value = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        value += lhs[i] * rhs[i];
    }
    return value;
}

static double norm(const std::vector<double>& values) {
    return std::sqrt(dot(values, values));
}

QRDecomposition modified_gram_schmidt(const Matrix& matrix) {
    const std::size_t m = matrix.size();
    const std::size_t n = matrix.front().size();
    for (const auto& row : matrix) {
        if (row.size() != n) {
            throw std::invalid_argument("matrix must be rectangular");
        }
    }

    Matrix q(m, std::vector<double>(n, 0.0));
    Matrix r(n, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> columns(n, std::vector<double>(m, 0.0));
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < m; ++i) {
            columns[j][i] = matrix[i][j];
        }
    }

    for (std::size_t j = 0; j < n; ++j) {
        std::vector<double> v = columns[j];
        for (std::size_t i = 0; i < j; ++i) {
            std::vector<double> qi(m, 0.0);
            for (std::size_t row = 0; row < m; ++row) {
                qi[row] = q[row][i];
            }
            r[i][j] = dot(qi, v);
            for (std::size_t row = 0; row < m; ++row) {
                v[row] -= r[i][j] * q[row][i];
            }
        }
        r[j][j] = norm(v);
        if (r[j][j] < 1e-12) {
            throw std::runtime_error("matrix has linearly dependent columns");
        }
        for (std::size_t row = 0; row < m; ++row) {
            q[row][j] = v[row] / r[j][j];
        }
    }

    return {q, r};
}

int main() {
    Matrix matrix{{1.0, 1.0}, {1.0, 0.0}, {0.0, 1.0}};
    const auto qr = modified_gram_schmidt(matrix);
    std::cout << "Q\n";
    for (const auto& row : qr.q) {
        for (double value : row) {
            std::cout << std::setw(12) << value;
        }
        std::cout << '\n';
    }
    std::cout << "R\n";
    for (const auto& row : qr.r) {
        for (double value : row) {
            std::cout << std::setw(12) << value;
        }
        std::cout << '\n';
    }
    return 0;
}
