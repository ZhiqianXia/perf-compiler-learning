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

static double dot(const Vector& lhs, const Vector& rhs) {
    double result = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) result += lhs[i] * rhs[i];
    return result;
}

Vector conjugate_gradient(const Matrix& a, const Vector& b, double tolerance, int max_iterations) {
    Vector x(b.size(), 0.0);
    Vector r = b;
    Vector p = r;
    double rsold = dot(r, r);
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        const Vector ap = matvec(a, p);
        const double alpha = rsold / dot(p, ap);
        for (std::size_t i = 0; i < x.size(); ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        const double rsnew = dot(r, r);
        if (std::sqrt(rsnew) < tolerance) {
            return x;
        }
        for (std::size_t i = 0; i < p.size(); ++i) {
            p[i] = r[i] + (rsnew / rsold) * p[i];
        }
        rsold = rsnew;
    }
    throw std::runtime_error("conjugate gradient did not converge");
}

int main() {
    Matrix a{{4.0, 1.0}, {1.0, 3.0}};
    Vector b{1.0, 2.0};
    const auto x = conjugate_gradient(a, b, 1e-12, 20);
    for (double value : x) std::cout << value << '\n';
    return 0;
}
