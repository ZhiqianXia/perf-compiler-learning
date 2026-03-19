#include <complex>
#include <iostream>
#include <stdexcept>

std::complex<double> robust_complex_divide(const std::complex<double>& lhs, const std::complex<double>& rhs) {
    const double a = lhs.real();
    const double b = lhs.imag();
    const double c = rhs.real();
    const double d = rhs.imag();
    if (c == 0.0 && d == 0.0) {
        throw std::invalid_argument("division by zero complex number");
    }
    if (std::abs(c) >= std::abs(d)) {
        const double ratio = d / c;
        const double denom = c + d * ratio;
        return {(a + b * ratio) / denom, (b - a * ratio) / denom};
    }
    const double ratio = c / d;
    const double denom = c * ratio + d;
    return {(a * ratio + b) / denom, (b * ratio - a) / denom};
}

int main() {
    const std::complex<double> lhs(3.0, 2.0);
    const std::complex<double> rhs(4.0, -5.0);
    std::cout << robust_complex_divide(lhs, rhs) << '\n';
    return 0;
}
