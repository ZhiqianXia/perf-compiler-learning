#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using Matrix = std::vector<std::vector<double>>;

Matrix hessenberg_reduction(Matrix matrix) {
    const std::size_t n = matrix.size();
    for (const auto& row : matrix) {
        if (row.size() != n) {
            throw std::invalid_argument("matrix must be square");
        }
    }

    for (std::size_t k = 0; k + 2 < n; ++k) {
        double norm = 0.0;
        for (std::size_t i = k + 1; i < n; ++i) {
            norm += matrix[i][k] * matrix[i][k];
        }
        norm = std::sqrt(norm);
        if (norm < 1e-12) {
            continue;
        }
        std::vector<double> v(n, 0.0);
        v[k + 1] = matrix[k + 1][k] + (matrix[k + 1][k] >= 0.0 ? norm : -norm);
        for (std::size_t i = k + 2; i < n; ++i) {
            v[i] = matrix[i][k];
        }
        double beta = 0.0;
        for (double value : v) {
            beta += value * value;
        }
        if (beta < 1e-12) {
            continue;
        }
        beta = 2.0 / beta;

        for (std::size_t j = k; j < n; ++j) {
            double dot = 0.0;
            for (std::size_t i = k + 1; i < n; ++i) {
                dot += v[i] * matrix[i][j];
            }
            for (std::size_t i = k + 1; i < n; ++i) {
                matrix[i][j] -= beta * v[i] * dot;
            }
        }
        for (std::size_t i = 0; i < n; ++i) {
            double dot = 0.0;
            for (std::size_t j = k + 1; j < n; ++j) {
                dot += matrix[i][j] * v[j];
            }
            for (std::size_t j = k + 1; j < n; ++j) {
                matrix[i][j] -= beta * dot * v[j];
            }
        }
    }
    return matrix;
}

int main() {
    Matrix matrix{{4.0, 1.0, -2.0, 2.0}, {1.0, 2.0, 0.0, 1.0}, {-2.0, 0.0, 3.0, -2.0}, {2.0, 1.0, -2.0, -1.0}};
    const Matrix h = hessenberg_reduction(matrix);
    for (const auto& row : h) {
        for (double value : row) {
            std::cout << std::setw(12) << value;
        }
        std::cout << '\n';
    }
    return 0;
}
