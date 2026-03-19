#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

using Matrix = std::vector<std::vector<double>>;

struct LUDecomposition {
    Matrix lower;
    Matrix upper;
};

LUDecomposition doolittle_lu(const Matrix& matrix) {
    const std::size_t n = matrix.size();
    Matrix lower(n, std::vector<double>(n, 0.0));
    Matrix upper(n, std::vector<double>(n, 0.0));

    for (const auto& row : matrix) {
        if (row.size() != n) {
            throw std::invalid_argument("matrix must be square");
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = i; k < n; ++k) {
            double sum = 0.0;
            for (std::size_t j = 0; j < i; ++j) {
                sum += lower[i][j] * upper[j][k];
            }
            upper[i][k] = matrix[i][k] - sum;
        }

        lower[i][i] = 1.0;
        for (std::size_t k = i + 1; k < n; ++k) {
            double sum = 0.0;
            for (std::size_t j = 0; j < i; ++j) {
                sum += lower[k][j] * upper[j][i];
            }
            if (std::abs(upper[i][i]) < 1e-12) {
                throw std::runtime_error("zero pivot encountered in LU decomposition");
            }
            lower[k][i] = (matrix[k][i] - sum) / upper[i][i];
        }
    }

    return {lower, upper};
}

int main() {
    Matrix matrix{{2.0, -1.0, -2.0}, {-4.0, 6.0, 3.0}, {-4.0, -2.0, 8.0}};
    const auto lu = doolittle_lu(matrix);
    std::cout << "L\n";
    for (const auto& row : lu.lower) {
        for (double value : row) std::cout << std::setw(10) << value;
        std::cout << '\n';
    }
    std::cout << "U\n";
    for (const auto& row : lu.upper) {
        for (double value : row) std::cout << std::setw(10) << value;
        std::cout << '\n';
    }
    return 0;
}
