#include <iostream>
#include <stdexcept>
#include <vector>

struct SplineSegment {
    double a;
    double b;
    double c;
    double d;
    double x_left;
};

std::vector<SplineSegment> natural_cubic_spline(const std::vector<double>& xs,
                                                const std::vector<double>& ys) {
    const std::size_t n = xs.size();
    if (n != ys.size() || n < 2) {
        throw std::invalid_argument("invalid spline input");
    }

    std::vector<double> h(n - 1, 0.0);
    for (std::size_t i = 0; i + 1 < n; ++i) {
        h[i] = xs[i + 1] - xs[i];
    }

    std::vector<double> alpha(n, 0.0), lower(n, 0.0), diagonal(n, 1.0), upper(n, 0.0), z(n, 0.0), c(n, 0.0), b(n - 1, 0.0), d(n - 1, 0.0);
    for (std::size_t i = 1; i + 1 < n; ++i) {
        alpha[i] = 3.0 * (ys[i + 1] - ys[i]) / h[i] - 3.0 * (ys[i] - ys[i - 1]) / h[i - 1];
    }
    for (std::size_t i = 1; i + 1 < n; ++i) {
        lower[i] = 2.0 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * upper[i - 1];
        upper[i] = h[i] / lower[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / lower[i];
    }
    for (int j = static_cast<int>(n) - 2; j >= 0; --j) {
        c[j] = z[j] - upper[j] * c[j + 1];
        b[j] = (ys[j + 1] - ys[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
    }

    std::vector<SplineSegment> segments;
    for (std::size_t i = 0; i + 1 < n; ++i) {
        segments.push_back({ys[i], b[i], c[i], d[i], xs[i]});
    }
    return segments;
}

double evaluate_spline(const std::vector<SplineSegment>& segments, double x) {
    std::size_t index = segments.size() - 1;
    for (std::size_t i = 0; i < segments.size(); ++i) {
        if (x >= segments[i].x_left) {
            index = i;
        }
    }
    const auto& segment = segments[index];
    const double dx = x - segment.x_left;
    return segment.a + segment.b * dx + segment.c * dx * dx + segment.d * dx * dx * dx;
}

int main() {
    const auto spline = natural_cubic_spline({0.0, 1.0, 2.0, 3.0}, {0.0, 1.0, 0.0, 1.0});
    std::cout << evaluate_spline(spline, 1.5) << '\n';
    return 0;
}
