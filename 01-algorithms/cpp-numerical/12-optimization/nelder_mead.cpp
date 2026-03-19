#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

using Point = std::vector<double>;

struct Vertex {
    Point point;
    double value;
};

Point centroid_excluding_worst(const std::vector<Vertex>& simplex) {
    const std::size_t dimension = simplex.front().point.size();
    Point center(dimension, 0.0);
    for (std::size_t i = 0; i + 1 < simplex.size(); ++i) {
        for (std::size_t j = 0; j < dimension; ++j) {
            center[j] += simplex[i].point[j];
        }
    }
    for (double& value : center) {
        value /= static_cast<double>(simplex.size() - 1);
    }
    return center;
}

Point affine_combination(const Point& a, const Point& b, double alpha, double beta) {
    Point result(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i] = alpha * a[i] + beta * b[i];
    }
    return result;
}

Vertex nelder_mead(const std::function<double(const Point&)>& objective,
                   std::vector<Point> initial_simplex,
                   double tolerance,
                   int max_iterations) {
    std::vector<Vertex> simplex;
    for (const auto& point : initial_simplex) {
        simplex.push_back({point, objective(point)});
    }

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        std::sort(simplex.begin(), simplex.end(),
                  [](const Vertex& lhs, const Vertex& rhs) { return lhs.value < rhs.value; });
        double variance = 0.0;
        double mean = 0.0;
        for (const auto& vertex : simplex) {
            mean += vertex.value / static_cast<double>(simplex.size());
        }
        for (const auto& vertex : simplex) {
            variance += (vertex.value - mean) * (vertex.value - mean);
        }
        variance /= static_cast<double>(simplex.size());
        if (std::sqrt(variance) < tolerance) {
            return simplex.front();
        }

        const Point center = centroid_excluding_worst(simplex);
        const Point reflected = affine_combination(center, simplex.back().point, 2.0, -1.0);
        const double reflected_value = objective(reflected);

        if (reflected_value < simplex.front().value) {
            const Point expanded = affine_combination(center, reflected, -1.0, 2.0);
            const double expanded_value = objective(expanded);
            simplex.back() = expanded_value < reflected_value ? Vertex{expanded, expanded_value}
                                                              : Vertex{reflected, reflected_value};
            continue;
        }

        if (reflected_value < simplex[simplex.size() - 2].value) {
            simplex.back() = {reflected, reflected_value};
            continue;
        }

        const Point contracted = affine_combination(center, simplex.back().point, 0.5, 0.5);
        const double contracted_value = objective(contracted);
        if (contracted_value < simplex.back().value) {
            simplex.back() = {contracted, contracted_value};
            continue;
        }

        for (std::size_t i = 1; i < simplex.size(); ++i) {
            simplex[i].point = affine_combination(simplex.front().point, simplex[i].point, 0.5, 0.5);
            simplex[i].value = objective(simplex[i].point);
        }
    }
    throw std::runtime_error("nelder-mead did not converge");
}

int main() {
    auto rosenbrock = [](const Point& point) {
        const double x = point[0];
        const double y = point[1];
        return std::pow(1.0 - x, 2) + 100.0 * std::pow(y - x * x, 2);
    };
    const auto result = nelder_mead(rosenbrock, {{-1.2, 1.0}, {0.0, 0.0}, {1.2, 1.2}}, 1e-8, 300);
    std::cout << result.point[0] << " " << result.point[1] << " " << result.value << '\n';
    return 0;
}
