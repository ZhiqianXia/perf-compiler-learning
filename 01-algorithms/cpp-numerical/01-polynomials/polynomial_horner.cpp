#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

class Polynomial {
public:
    explicit Polynomial(std::vector<double> coefficients)
        : coefficients_(std::move(coefficients)) {
        if (coefficients_.empty()) {
            throw std::invalid_argument("coefficients must not be empty");
        }
    }

    double evaluate(double x) const {
        double value = 0.0;
        for (double coefficient : coefficients_) {
            value = value * x + coefficient;
        }
        return value;
    }

    std::pair<double, double> evaluate_with_derivative(double x) const {
        double value = coefficients_.front();
        double derivative = 0.0;
        for (std::size_t i = 1; i < coefficients_.size(); ++i) {
            derivative = derivative * x + value;
            value = value * x + coefficients_[i];
        }
        return {value, derivative};
    }

private:
    std::vector<double> coefficients_;
};

int main() {
    Polynomial polynomial({2.0, -6.0, 2.0, -1.0});
    const double x = 3.0;
    const auto [value, derivative] = polynomial.evaluate_with_derivative(x);
    std::cout << "p(" << x << ") = " << value << '\n';
    std::cout << "p'(" << x << ") = " << derivative << '\n';
    return 0;
}
