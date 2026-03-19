#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using Matrix = std::vector<std::vector<double>>;

std::vector<double> solve_linear_system(Matrix matrix, std::vector<double> rhs) {
    const std::size_t n = matrix.size();
    if (rhs.size() != n) {
        throw std::invalid_argument("rhs size mismatch");
    }
    for (const auto& row : matrix) {
        if (row.size() != n) {
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
            throw std::runtime_error("matrix is singular or ill-conditioned");
        }
        std::swap(matrix[pivot], matrix[pivot_row]);
        std::swap(rhs[pivot], rhs[pivot_row]);

        for (std::size_t row = pivot + 1; row < n; ++row) {
            const double factor = matrix[row][pivot] / matrix[pivot][pivot];
            for (std::size_t column = pivot; column < n; ++column) {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }

    std::vector<double> solution(n, 0.0);
    for (int row = static_cast<int>(n) - 1; row >= 0; --row) {
        double sum = rhs[row];
        for (std::size_t column = row + 1; column < n; ++column) {
            sum -= matrix[row][column] * solution[column];
        }
        solution[row] = sum / matrix[row][row];
    }
    return solution;
}

int main() {
    Matrix matrix{{2.0, 1.0, -1.0}, {-3.0, -1.0, 2.0}, {-2.0, 1.0, 2.0}};
    std::vector<double> rhs{8.0, -11.0, -3.0};
    const auto solution = solve_linear_system(matrix, rhs);
    for (double value : solution) {
        std::cout << value << '\n';
    }
    return 0;
}
