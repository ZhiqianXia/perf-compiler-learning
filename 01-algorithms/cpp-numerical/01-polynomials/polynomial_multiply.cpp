#include <iostream>
#include <vector>

std::vector<double> polynomial_multiply(const std::vector<double>& lhs, const std::vector<double>& rhs) {
    std::vector<double> result(lhs.size() + rhs.size() - 1, 0.0);
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        for (std::size_t j = 0; j < rhs.size(); ++j) {
            result[i + j] += lhs[i] * rhs[j];
        }
    }
    return result;
}

int main() {
    const auto result = polynomial_multiply({1.0, 2.0, 1.0}, {1.0, -1.0});
    for (double value : result) {
        std::cout << value << ' ';
    }
    std::cout << '\n';
    return 0;
}
