#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

struct DivisionResult {
    std::vector<double> quotient;
    std::vector<double> remainder;
};

DivisionResult polynomial_long_division(std::vector<double> dividend, const std::vector<double>& divisor) {
    if (divisor.empty() || divisor.front() == 0.0) {
        throw std::invalid_argument("divisor must have nonzero leading coefficient");
    }
    if (dividend.size() < divisor.size()) {
        return {{0.0}, dividend};
    }

    std::vector<double> quotient(dividend.size() - divisor.size() + 1, 0.0);
    for (std::size_t i = 0; i < quotient.size(); ++i) {
        const double scale = dividend[i] / divisor.front();
        quotient[i] = scale;
        for (std::size_t j = 0; j < divisor.size(); ++j) {
            dividend[i + j] -= scale * divisor[j];
        }
    }
    std::vector<double> remainder(dividend.end() - static_cast<long>(divisor.size() - 1), dividend.end());
    return {quotient, remainder};
}

int main() {
    const auto result = polynomial_long_division({1.0, -3.0, 0.0, 4.0}, {1.0, -1.0});
    std::cout << "quotient: ";
    for (double value : result.quotient) std::cout << value << ' ';
    std::cout << "\nremainder: ";
    for (double value : result.remainder) std::cout << value << ' ';
    std::cout << '\n';
    return 0;
}
