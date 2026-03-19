#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>

double composite_trapezoidal(const std::function<double(double)>& f,
                             double left,
                             double right,
                             int subintervals) {
    if (subintervals <= 0) {
        throw std::invalid_argument("subintervals must be positive");
    }
    const double h = (right - left) / static_cast<double>(subintervals);
    double sum = 0.5 * (f(left) + f(right));
    for (int i = 1; i < subintervals; ++i) {
        sum += f(left + i * h);
    }
    return h * sum;
}

int main() {
    auto f = [](double x) { return std::cos(x); };
    std::cout << composite_trapezoidal(f, 0.0, 1.0, 1000) << '\n';
    return 0;
}
