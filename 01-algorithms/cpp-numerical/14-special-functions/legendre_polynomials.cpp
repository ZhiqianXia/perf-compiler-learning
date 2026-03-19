#include <iostream>
#include <vector>

std::vector<double> legendre_values(int max_degree, double x) {
    std::vector<double> values(max_degree + 1, 0.0);
    values[0] = 1.0;
    if (max_degree >= 1) {
        values[1] = x;
    }
    for (int n = 1; n < max_degree; ++n) {
        values[n + 1] = ((2.0 * n + 1.0) * x * values[n] - n * values[n - 1]) / (n + 1.0);
    }
    return values;
}

int main() {
    const auto values = legendre_values(5, 0.3);
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::cout << "P_" << i << " = " << values[i] << '\n';
    }
    return 0;
}
