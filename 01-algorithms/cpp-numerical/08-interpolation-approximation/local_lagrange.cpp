#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

using Point = std::pair<double, double>;

static double lagrange_three_point(const std::vector<Point>& points, double x) {
    double result = 0.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        double basis = 1.0;
        for (std::size_t j = 0; j < points.size(); ++j) {
            if (i != j) {
                basis *= (x - points[j].first) / (points[i].first - points[j].first);
            }
        }
        result += basis * points[i].second;
    }
    return result;
}

double local_lagrange_interpolate(const std::vector<double>& xs,
                                  const std::vector<double>& ys,
                                  double x) {
    if (xs.size() != ys.size() || xs.size() < 3) {
        throw std::invalid_argument("need at least three sample points");
    }
    std::vector<Point> points;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        points.push_back({xs[i], ys[i]});
    }
    std::nth_element(points.begin(), points.begin() + 2, points.end(),
                     [x](const Point& lhs, const Point& rhs) {
                         return std::abs(lhs.first - x) < std::abs(rhs.first - x);
                     });
    points.resize(3);
    return lagrange_three_point(points, x);
}

int main() {
    std::vector<double> xs{0.0, 0.5, 1.0, 1.5, 2.0};
    std::vector<double> ys{1.0, 1.64872, 2.71828, 4.48169, 7.38906};
    std::cout << local_lagrange_interpolate(xs, ys, 1.2) << '\n';
    return 0;
}
