#include <cmath>
#include <complex>
#include <iostream>

std::complex<double> complex_log_manual(const std::complex<double>& z) {
    const double magnitude = std::abs(z);
    const double angle = std::atan2(z.imag(), z.real());
    return {std::log(magnitude), angle};
}

int main() {
    const std::complex<double> z(2.0, 3.0);
    const auto manual = complex_log_manual(z);
    const auto standard = std::log(z);
    std::cout << "manual  = " << manual << '\n';
    std::cout << "std::log = " << standard << '\n';
    std::cout << "error   = " << std::abs(manual - standard) << '\n';
    return 0;
}
