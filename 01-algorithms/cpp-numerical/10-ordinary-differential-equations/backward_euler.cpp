#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>

// Scalar backward Euler with Newton iterations.
double backward_euler_step(const std::function<double(double, double)>& f,
                           const std::function<double(double, double)>& df_dy,
                           double t,
                           double y,
                           double h,
                           int newton_iterations = 20) {
    double next = y;
    for (int iteration = 0; iteration < newton_iterations; ++iteration) {
        const double residual = next - y - h * f(t + h, next);
        const double jacobian = 1.0 - h * df_dy(t + h, next);
        if (std::abs(jacobian) < 1e-14) {
            throw std::runtime_error("backward euler newton jacobian too small");
        }
        const double updated = next - residual / jacobian;
        if (std::abs(updated - next) < 1e-12) {
            return updated;
        }
        next = updated;
    }
    throw std::runtime_error("backward euler failed to converge");
}

int main() {
    auto f = [](double, double y) { return -10.0 * y; };
    auto df = [](double, double) { return -10.0; };
    double t = 0.0;
    double y = 1.0;
    const double h = 0.1;
    for (int step = 0; step < 10; ++step) {
        std::cout << t << " " << y << '\n';
        y = backward_euler_step(f, df, t, y, h);
        t += h;
    }
    return 0;
}
