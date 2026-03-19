#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

double neville_interpolate(const std::vector<double>& xs,
                           const std::vector<double>& ys,
                           double x) {
    if (xs.size() != ys.size() || xs.empty()) {
        throw std::invalid_argument("input size mismatch");
    }
    std::vector<double> table = ys;
    for (std::size_t j = 1; j < xs.size(); ++j) {
        for (std::size_t i = 0; i < xs.size() - j; ++i) {
            table[i] = ((x - xs[i + j]) * table[i] + (xs[i] - x) * table[i + 1]) / (xs[i] - xs[i + j]);
        }
    }
    return table[0];
}

int main() {
    std::vector<double> xs{0.0, 0.25, 0.5, 0.75, 1.0};
    std::vector<double> ys;
    for (double x : xs) {
        ys.push_back(std::exp(x));
    }
    std::cout << neville_interpolate(xs, ys, 0.4) << '\n';
    return 0;
}
