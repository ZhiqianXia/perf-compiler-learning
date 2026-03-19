#include <cmath>
#include <functional>
#include <iostream>
#include <random>

double monte_carlo_integral(const std::function<double(double)>& f,
                            double left,
                            double right,
                            int samples,
                            unsigned seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribution(left, right);
    double sum = 0.0;
    for (int i = 0; i < samples; ++i) {
        sum += f(distribution(generator));
    }
    return (right - left) * sum / static_cast<double>(samples);
}

int main() {
    auto f = [](double x) { return std::sqrt(1.0 - x * x); };
    std::cout << monte_carlo_integral(f, 0.0, 1.0, 200000, 2026) << '\n';
    return 0;
}
