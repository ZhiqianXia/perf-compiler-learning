#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <random>
#include <utility>

std::pair<double, double> box_muller(std::mt19937& generator) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double u1 = 0.0;
    while (u1 <= 0.0) {
        u1 = distribution(generator);
    }
    const double u2 = distribution(generator);
    const double radius = std::sqrt(-2.0 * std::log(u1));
    const double theta = 2.0 * std::numbers::pi * u2;
    return {radius * std::cos(theta), radius * std::sin(theta)};
}

int main() {
    std::mt19937 generator(42);
    for (int i = 0; i < 5; ++i) {
        const auto [z1, z2] = box_muller(generator);
        std::cout << std::fixed << std::setprecision(6) << z1 << ", " << z2 << '\n';
    }
    return 0;
}
