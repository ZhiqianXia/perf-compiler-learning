#include <cmath>
#include <functional>
#include <iostream>

// Three-point Gauss-Legendre quadrature on [a, b].
double gauss_legendre_3(const std::function<double(double)>& f, double a, double b) {
    static const double nodes[3] = {-0.7745966692414834, 0.0, 0.7745966692414834};
    static const double weights[3] = {0.5555555555555556, 0.8888888888888888, 0.5555555555555556};
    const double center = 0.5 * (a + b);
    const double half_length = 0.5 * (b - a);
    double result = 0.0;
    for (int i = 0; i < 3; ++i) {
        result += weights[i] * f(center + half_length * nodes[i]);
    }
    return half_length * result;
}

int main() {
    auto f = [](double x) { return std::log(1.0 + x); };
    std::cout << gauss_legendre_3(f, 0.0, 1.0) << '\n';
    return 0;
}
