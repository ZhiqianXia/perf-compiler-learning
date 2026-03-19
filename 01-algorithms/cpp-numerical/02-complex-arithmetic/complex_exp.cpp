#include <cmath>
#include <complex>
#include <iostream>

std::complex<double> complex_exp_manual(const std::complex<double>& z) {
    const double magnitude = std::exp(z.real());
    return {magnitude * std::cos(z.imag()), magnitude * std::sin(z.imag())};
}

int main() {
    const std::complex<double> z(1.2, 0.7);
    const auto manual = complex_exp_manual(z);
    const auto standard = std::exp(z);

    std::cout << "manual  = " << manual << '\n';
    std::cout << "std::exp = " << standard << '\n';
    std::cout << "error   = " << std::abs(manual - standard) << '\n';
    return 0;
}
