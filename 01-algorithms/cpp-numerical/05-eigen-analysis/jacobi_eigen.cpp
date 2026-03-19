#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using Matrix = std::vector<std::vector<double>>;

std::vector<double> jacobi_eigenvalues(Matrix matrix, double tolerance = 1e-10, int max_iterations = 100) {
    const std::size_t n = matrix.size();
    for (const auto& row : matrix) {
        if (row.size() != n) {
            throw std::invalid_argument("matrix must be square");
        }
    }

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        std::size_t p = 0;
        std::size_t q = 1;
        double max_value = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                if (std::abs(matrix[i][j]) > max_value) {
                    max_value = std::abs(matrix[i][j]);
                    p = i;
                    q = j;
                }
            }
        }
        if (max_value < tolerance) {
            break;
        }

        const double theta = 0.5 * std::atan2(2.0 * matrix[p][q], matrix[q][q] - matrix[p][p]);
        const double cosine = std::cos(theta);
        const double sine = std::sin(theta);

        const double app = matrix[p][p];
        const double aqq = matrix[q][q];
        const double apq = matrix[p][q];
        matrix[p][p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
        matrix[q][q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;

        for (std::size_t r = 0; r < n; ++r) {
            if (r == p || r == q) {
                continue;
            }
            const double arp = matrix[r][p];
            const double arq = matrix[r][q];
            matrix[r][p] = cosine * arp - sine * arq;
            matrix[p][r] = matrix[r][p];
            matrix[r][q] = sine * arp + cosine * arq;
            matrix[q][r] = matrix[r][q];
        }
    }

    std::vector<double> eigenvalues(n);
    for (std::size_t i = 0; i < n; ++i) {
        eigenvalues[i] = matrix[i][i];
    }
    return eigenvalues;
}

int main() {
    Matrix matrix{{4.0, -2.0, 2.0}, {-2.0, 1.0, 0.0}, {2.0, 0.0, 3.0}};
    const auto eigenvalues = jacobi_eigenvalues(matrix);
    for (double value : eigenvalues) {
        std::cout << std::setw(12) << value << '\n';
    }
    return 0;
}
