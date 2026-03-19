#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

using State = std::vector<double>;
using ODEFunction = std::function<State(double, const State&)>;

State add_scaled(const State& state, const State& direction, double scale) {
    State result(state.size());
    for (std::size_t i = 0; i < state.size(); ++i) {
        result[i] = state[i] + scale * direction[i];
    }
    return result;
}

State improved_euler_step(const ODEFunction& f, double t, const State& y, double h) {
    const State k1 = f(t, y);
    const State predictor = add_scaled(y, k1, h);
    const State k2 = f(t + h, predictor);
    State next(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        next[i] = y[i] + 0.5 * h * (k1[i] + k2[i]);
    }
    return next;
}

std::vector<State> adams_bashforth_moulton4(const ODEFunction& f,
                                            double t0,
                                            const State& y0,
                                            double h,
                                            int steps) {
    if (steps < 4) {
        throw std::invalid_argument("steps must be at least 4");
    }
    std::vector<State> solution;
    solution.push_back(y0);
    double t = t0;
    for (int i = 0; i < 3; ++i) {
        solution.push_back(improved_euler_step(f, t, solution.back(), h));
        t += h;
    }

    std::vector<State> derivatives;
    for (int i = 0; i < 4; ++i) {
        derivatives.push_back(f(t0 + i * h, solution[i]));
    }

    for (int step = 3; step < steps - 1; ++step) {
        const State& yn = solution[step];
        State predictor(yn.size());
        for (std::size_t i = 0; i < yn.size(); ++i) {
            predictor[i] = yn[i] + h * (55.0 * derivatives[3][i] - 59.0 * derivatives[2][i] +
                                        37.0 * derivatives[1][i] - 9.0 * derivatives[0][i]) / 24.0;
        }
        const State predictor_derivative = f(t0 + (step + 1) * h, predictor);
        State corrector(yn.size());
        for (std::size_t i = 0; i < yn.size(); ++i) {
            corrector[i] = yn[i] + h * (9.0 * predictor_derivative[i] + 19.0 * derivatives[3][i] -
                                        5.0 * derivatives[2][i] + derivatives[1][i]) / 24.0;
        }
        solution.push_back(corrector);
        derivatives.erase(derivatives.begin());
        derivatives.push_back(f(t0 + (step + 1) * h, corrector));
    }
    return solution;
}

int main() {
    ODEFunction f = [](double, const State& y) { return State{-y[0]}; };
    const auto solution = adams_bashforth_moulton4(f, 0.0, State{1.0}, 0.1, 8);
    for (std::size_t i = 0; i < solution.size(); ++i) {
        std::cout << i * 0.1 << " " << solution[i][0] << '\n';
    }
    return 0;
}
