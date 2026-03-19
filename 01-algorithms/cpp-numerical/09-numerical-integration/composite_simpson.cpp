#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>

double composite_simpson(const std::function<double(double)>& f,
                         double left,
                         double right,
                         int subintervals) {
    if (subintervals <= 0 || subintervals % 2 != 0) {
        throw std::invalid_argument("subintervals must be a positive even number");
    }
    const double h = (right - left) / static_cast<double>(subintervals);
    double sum = f(left) + f(right);
    for (int i = 1; i < subintervals; ++i) {
        const double x = left + i * h;
        sum += (i % 2 == 0 ? 2.0 : 4.0) * f(x);
    }
    return h * sum / 3.0;
}

int main() {
    auto f = [](double x) { return std::exp(-x * x); };
    std::cout << composite_simpson(f, 0.0, 1.0, 100) << '\n';
    return 0;
}
