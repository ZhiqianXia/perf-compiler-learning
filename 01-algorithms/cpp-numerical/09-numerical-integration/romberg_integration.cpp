#include <cmath>
#include <functional>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <vector>

double romberg_integral(const std::function<double(double)>& f,
                        double left,
                        double right,
                        double tolerance,
                        int max_levels) {
    std::vector<std::vector<double>> table(max_levels, std::vector<double>(max_levels, 0.0));
    double h = right - left;
    table[0][0] = 0.5 * h * (f(left) + f(right));

    for (int level = 1; level < max_levels; ++level) {
        h *= 0.5;
        double midpoint_sum = 0.0;
        const int count = 1 << (level - 1);
        for (int k = 1; k <= count; ++k) {
            midpoint_sum += f(left + (2 * k - 1) * h);
        }
        table[level][0] = 0.5 * table[level - 1][0] + h * midpoint_sum;
        for (int column = 1; column <= level; ++column) {
            const double factor = std::pow(4.0, column);
            table[level][column] = table[level][column - 1] +
                                   (table[level][column - 1] - table[level - 1][column - 1]) /
                                       (factor - 1.0);
        }
        if (std::abs(table[level][level] - table[level - 1][level - 1]) < tolerance) {
            return table[level][level];
        }
    }
    throw std::runtime_error("romberg integration did not converge");
}

int main() {
    auto f = [](double x) { return std::sin(x); };
    std::cout << romberg_integral(f, 0.0, std::numbers::pi, 1e-10, 10) << '\n';
    return 0;
}
