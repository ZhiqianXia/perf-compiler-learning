#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>

struct RootResult {
    double value;
    int iterations;
};

RootResult newton_method(const std::function<double(double)>& f,
                         const std::function<double(double)>& derivative,
                         double initial_guess,
                         double tolerance,
                         int max_iterations) {
    double x = initial_guess;
    for (int iteration = 1; iteration <= max_iterations; ++iteration) {
        const double fx = f(x);
        const double dfx = derivative(x);
        if (std::abs(dfx) < 1e-14) {
            throw std::runtime_error("derivative too small");
        }
        const double next = x - fx / dfx;
        if (std::abs(next - x) < tolerance) {
            return {next, iteration};
        }
        x = next;
    }
    throw std::runtime_error("newton method did not converge");
}

RootResult secant_method(const std::function<double(double)>& f,
                         double x0,
                         double x1,
                         double tolerance,
                         int max_iterations) {
    for (int iteration = 1; iteration <= max_iterations; ++iteration) {
        const double f0 = f(x0);
        const double f1 = f(x1);
        const double denominator = f1 - f0;
        if (std::abs(denominator) < 1e-14) {
            throw std::runtime_error("secant denominator too small");
        }
        const double x2 = x1 - f1 * (x1 - x0) / denominator;
        if (std::abs(x2 - x1) < tolerance) {
            return {x2, iteration};
        }
        x0 = x1;
        x1 = x2;
    }
    throw std::runtime_error("secant method did not converge");
}

int main() {
    auto f = [](double x) { return x * x * x - x - 2.0; };
    auto df = [](double x) { return 3.0 * x * x - 1.0; };
    const auto newton = newton_method(f, df, 1.5, 1e-12, 30);
    const auto secant = secant_method(f, 1.0, 2.0, 1e-12, 30);
    std::cout << "newton: " << newton.value << " in " << newton.iterations << " steps\n";
    std::cout << "secant: " << secant.value << " in " << secant.iterations << " steps\n";
    return 0;
}
