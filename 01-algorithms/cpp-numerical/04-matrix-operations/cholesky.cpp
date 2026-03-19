#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using Matrix = std::vector<std::vector<double>>;

Matrix cholesky_decompose(const Matrix& matrix) {
    const std::size_t n = matrix.size();
    Matrix lower(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        if (matrix[i].size() != n) {
            throw std::invalid_argument("matrix must be square");
        }
        for (std::size_t j = 0; j <= i; ++j) {
            double sum = matrix[i][j];
            for (std::size_t k = 0; k < j; ++k) {
                sum -= lower[i][k] * lower[j][k];
            }
            if (i == j) {
                if (sum <= 0.0) {
                    throw std::runtime_error("matrix is not symmetric positive definite");
                }
                lower[i][j] = std::sqrt(sum);
            } else {
                lower[i][j] = sum / lower[j][j];
            }
        }
    }
    return lower;
}

int main() {
    Matrix matrix{{4.0, 12.0, -16.0}, {12.0, 37.0, -43.0}, {-16.0, -43.0, 98.0}};
    const Matrix lower = cholesky_decompose(matrix);
    for (const auto& row : lower) {
        for (double value : row) {
            std::cout << std::setw(10) << value;
        }
        std::cout << '\n';
    }
    return 0;
}
