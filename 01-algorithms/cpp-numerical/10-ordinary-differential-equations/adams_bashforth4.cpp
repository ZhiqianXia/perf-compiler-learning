#include <functional>
#include <iostream>
#include <vector>

using State = std::vector<double>;
using ODEFunction = std::function<State(double, const State&)>;

static State add_scaled(const State& a, const State& b, double scale) {
    State result(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) result[i] = a[i] + scale * b[i];
    return result;
}

static State rk4_step(const ODEFunction& f, double t, const State& y, double h) {
    const State k1 = f(t, y);
    const State k2 = f(t + 0.5 * h, add_scaled(y, k1, 0.5 * h));
    const State k3 = f(t + 0.5 * h, add_scaled(y, k2, 0.5 * h));
    const State k4 = f(t + h, add_scaled(y, k3, h));
    State next(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        next[i] = y[i] + h * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    }
    return next;
}

int main() {
    ODEFunction f = [](double, const State& y) { return State{-y[0]}; };
    const double h = 0.1;
    std::vector<State> solution{State{1.0}};
    for (int i = 0; i < 3; ++i) {
        solution.push_back(rk4_step(f, i * h, solution.back(), h));
    }
    std::vector<State> derivatives;
    for (int i = 0; i < 4; ++i) derivatives.push_back(f(i * h, solution[i]));
    for (int step = 3; step < 10; ++step) {
        const State& yn = solution.back();
        State next(yn.size());
        for (std::size_t i = 0; i < yn.size(); ++i) {
            next[i] = yn[i] + h * (55.0 * derivatives[3][i] - 59.0 * derivatives[2][i] + 37.0 * derivatives[1][i] - 9.0 * derivatives[0][i]) / 24.0;
        }
        solution.push_back(next);
        derivatives.erase(derivatives.begin());
        derivatives.push_back(f((step + 1) * h, next));
    }
    for (std::size_t i = 0; i < solution.size(); ++i) {
        std::cout << i * h << " " << solution[i][0] << '\n';
    }
    return 0;
}
