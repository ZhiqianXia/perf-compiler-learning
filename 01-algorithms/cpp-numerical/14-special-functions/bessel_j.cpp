#include <cmath>
#include <iostream>
#include <stdexcept>

static double bessel_j0_series(double x) {
    double term = 1.0;
    double sum = 1.0;
    const double half_squared = x * x / 4.0;
    for (int k = 1; k < 30; ++k) {
        term *= -half_squared / (k * k);
        sum += term;
    }
    return sum;
}

static double bessel_j1_series(double x) {
    double term = x / 2.0;
    double sum = term;
    const double half_squared = x * x / 4.0;
    for (int k = 1; k < 30; ++k) {
        term *= -half_squared / (k * (k + 1.0));
        sum += term;
    }
    return sum;
}

double bessel_j(int n, double x) {
    if (n < 0) {
        throw std::invalid_argument("order must be non-negative");
    }
    if (n == 0) {
        return bessel_j0_series(x);
    }
    if (n == 1) {
        return bessel_j1_series(x);
    }
    if (x == 0.0) {
        return 0.0;
    }
    double j_prev = bessel_j0_series(x);
    double j_curr = bessel_j1_series(x);
    for (int k = 1; k < n; ++k) {
        const double j_next = 2.0 * k * j_curr / x - j_prev;
        j_prev = j_curr;
        j_curr = j_next;
    }
    return j_curr;
}

int main() {
    for (int n = 0; n <= 4; ++n) {
        std::cout << "J_" << n << "(1.5) = " << bessel_j(n, 1.5) << '\n';
    }
    return 0;
}
