#include <cmath>
#include <iostream>
#include <numbers>

double erf_approx(double x) {
    const double sign = x < 0.0 ? -1.0 : 1.0;
    x = std::abs(x);
    const double t = 1.0 / (1.0 + 0.3275911 * x);
    const double a1 = 0.254829592;
    const double a2 = -0.284496736;
    const double a3 = 1.421413741;
    const double a4 = -1.453152027;
    const double a5 = 1.061405429;
    const double poly = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t;
    return sign * (1.0 - poly * std::exp(-x * x));
}

double normal_cdf(double x) {
    return 0.5 * (1.0 + erf_approx(x / std::sqrt(2.0)));
}

int main() {
    for (double x : {-2.0, -1.0, 0.0, 1.0, 2.0}) {
        std::cout << x << " " << normal_cdf(x) << '\n';
    }
    return 0;
}
