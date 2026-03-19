#include <iostream>
#include <vector>
#include <functional>

using State = std::vector<double>;
using ODEFunction = std::function<State(double, const State&)>;

static State combine(const State& a, const State& b, double scale) {
    State result(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + scale * b[i];
    }
    return result;
}

State rk4_step(const ODEFunction& f, double t, const State& y, double h) {
    const State k1 = f(t, y);
    const State k2 = f(t + 0.5 * h, combine(y, k1, 0.5 * h));
    const State k3 = f(t + 0.5 * h, combine(y, k2, 0.5 * h));
    const State k4 = f(t + h, combine(y, k3, h));
    State next(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        next[i] = y[i] + h * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    }
    return next;
}

int main() {
    ODEFunction f = [](double, const State& y) { return State{y[0]}; };
    double t = 0.0;
    State y{1.0};
    const double h = 0.1;
    for (int step = 0; step < 10; ++step) {
        std::cout << t << " " << y[0] << '\n';
        y = rk4_step(f, t, y, h);
        t += h;
    }
    return 0;
}
