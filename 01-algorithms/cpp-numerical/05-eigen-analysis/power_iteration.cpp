#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

static Vector matvec(const Matrix& matrix, const Vector& vector) {
    Vector result(matrix.size(), 0.0);
    for (std::size_t i = 0; i < matrix.size(); ++i) {
        for (std::size_t j = 0; j < vector.size(); ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
    return result;
}

static double norm(const Vector& vector) {
    double value = 0.0;
    for (double x : vector) value += x * x;
    return std::sqrt(value);
}

int main() {
    Matrix matrix{{4.0, 1.0}, {2.0, 3.0}};
    Vector vector{1.0, 1.0};
    double eigenvalue = 0.0;
    for (int iteration = 0; iteration < 20; ++iteration) {
        Vector next = matvec(matrix, vector);
        const double scale = norm(next);
        for (double& x : next) x /= scale;
        Vector mv = matvec(matrix, next);
        double numerator = 0.0;
        double denominator = 0.0;
        for (std::size_t i = 0; i < next.size(); ++i) {
            numerator += next[i] * mv[i];
            denominator += next[i] * next[i];
        }
        eigenvalue = numerator / denominator;
        vector = next;
    }
    std::cout << "dominant eigenvalue = " << eigenvalue << '\n';
    return 0;
}
