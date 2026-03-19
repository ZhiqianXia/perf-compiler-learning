#include <cmath>
#include <iostream>
#include <stdexcept>

static double beta_continued_fraction(double a, double b, double x) {
    constexpr int max_iterations = 200;
    constexpr double epsilon = 1e-12;
    constexpr double tiny = 1e-30;

    double c = 1.0;
    double d = 1.0 - (a + b) * x / (a + 1.0);
    if (std::abs(d) < tiny) {
        d = tiny;
    }
    d = 1.0 / d;
    double h = d;

    for (int m = 1; m <= max_iterations; ++m) {
        const int m2 = 2 * m;
        double aa = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < tiny) {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if (std::abs(c) < tiny) {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0));
        d = 1.0 + aa * d;
        if (std::abs(d) < tiny) {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if (std::abs(c) < tiny) {
            c = tiny;
        }
        d = 1.0 / d;
        const double delta = d * c;
        h *= delta;
        if (std::abs(delta - 1.0) < epsilon) {
            break;
        }
    }
    return h;
}

double regularized_incomplete_beta(double a, double b, double x) {
    if (a <= 0.0 || b <= 0.0 || x < 0.0 || x > 1.0) {
        throw std::invalid_argument("invalid incomplete beta parameters");
    }
    if (x == 0.0) {
        return 0.0;
    }
    if (x == 1.0) {
        return 1.0;
    }

    const double log_factor = std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
                              a * std::log(x) + b * std::log(1.0 - x);
    const double factor = std::exp(log_factor);
    if (x < (a + 1.0) / (a + b + 2.0)) {
        return factor * beta_continued_fraction(a, b, x) / a;
    }
    return 1.0 - factor * beta_continued_fraction(b, a, 1.0 - x) / b;
}

int main() {
    std::cout << regularized_incomplete_beta(2.5, 3.0, 0.4) << '\n';
    return 0;
}
