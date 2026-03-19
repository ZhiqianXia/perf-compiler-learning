#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using Matrix = std::vector<std::vector<double>>;

Matrix invert_matrix(Matrix matrix) {
    const std::size_t n = matrix.size();
    Matrix inverse(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        inverse[i][i] = 1.0;
        if (matrix[i].size() != n) {
            throw std::invalid_argument("matrix must be square");
        }
    }

    for (std::size_t pivot = 0; pivot < n; ++pivot) {
        std::size_t pivot_row = pivot;
        for (std::size_t row = pivot + 1; row < n; ++row) {
            if (std::abs(matrix[row][pivot]) > std::abs(matrix[pivot_row][pivot])) {
                pivot_row = row;
            }
        }
        if (std::abs(matrix[pivot_row][pivot]) < 1e-12) {
            throw std::runtime_error("matrix is singular");
        }
        std::swap(matrix[pivot], matrix[pivot_row]);
        std::swap(inverse[pivot], inverse[pivot_row]);

        const double scale = matrix[pivot][pivot];
        for (std::size_t col = 0; col < n; ++col) {
            matrix[pivot][col] /= scale;
            inverse[pivot][col] /= scale;
        }

        for (std::size_t row = 0; row < n; ++row) {
            if (row == pivot) {
                continue;
            }
            const double factor = matrix[row][pivot];
            for (std::size_t col = 0; col < n; ++col) {
                matrix[row][col] -= factor * matrix[pivot][col];
                inverse[row][col] -= factor * inverse[pivot][col];
            }
        }
    }
    return inverse;
}

int main() {
    Matrix matrix{{4.0, 7.0}, {2.0, 6.0}};
    const Matrix inverse = invert_matrix(matrix);
    for (const auto& row : inverse) {
        for (double value : row) std::cout << std::setw(12) << value;
        std::cout << '\n';
    }
    return 0;
}
