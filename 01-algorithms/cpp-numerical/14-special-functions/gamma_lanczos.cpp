#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>

// Lanczos approximation for Gamma(x), x > 0.
double gamma_lanczos(double x) {
    static const double coefficients[] = {676.5203681218851, -1259.1392167224028, 771.3234287776531,
                                          -176.6150291621406, 12.507343278686905, -0.13857109526572012,
                                          9.984369578019572e-6, 1.5056327351493116e-7};
    if (x < 0.5) {
        return std::numbers::pi / (std::sin(std::numbers::pi * x) * gamma_lanczos(1.0 - x));
    }
    x -= 1.0;
    double a = 0.99999999999980993;
    for (int i = 0; i < 8; ++i) {
        a += coefficients[i] / (x + i + 1.0);
    }
    const double t = x + 7.5;
    return std::sqrt(2.0 * std::numbers::pi) * std::pow(t, x + 0.5) * std::exp(-t) * a;
}

int main() {
    std::cout << gamma_lanczos(0.5) << '\n';
    std::cout << gamma_lanczos(5.0) << '\n';
    return 0;
}
