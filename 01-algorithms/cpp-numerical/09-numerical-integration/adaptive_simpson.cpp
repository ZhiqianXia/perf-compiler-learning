#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>

static double simpson(const std::function<double(double)>& f, double left, double right) {
    const double mid = 0.5 * (left + right);
    return (right - left) * (f(left) + 4.0 * f(mid) + f(right)) / 6.0;
}

double adaptive_simpson_recursive(const std::function<double(double)>& f,
                                  double left,
                                  double right,
                                  double tolerance,
                                  double whole,
                                  int depth) {
    const double mid = 0.5 * (left + right);
    const double left_part = simpson(f, left, mid);
    const double right_part = simpson(f, mid, right);
    const double delta = left_part + right_part - whole;
    if (depth == 0) {
        throw std::runtime_error("adaptive simpson exceeded recursion depth");
    }
    if (std::abs(delta) <= 15.0 * tolerance) {
        return left_part + right_part + delta / 15.0;
    }
    return adaptive_simpson_recursive(f, left, mid, tolerance / 2.0, left_part, depth - 1) +
           adaptive_simpson_recursive(f, mid, right, tolerance / 2.0, right_part, depth - 1);
}

double adaptive_simpson(const std::function<double(double)>& f,
                        double left,
                        double right,
                        double tolerance,
                        int max_depth = 20) {
    return adaptive_simpson_recursive(f, left, right, tolerance, simpson(f, left, right), max_depth);
}

int main() {
    auto f = [](double x) { return 1.0 / (1.0 + x * x); };
    std::cout << adaptive_simpson(f, 0.0, 1.0, 1e-10) << '\n';
    return 0;
}
