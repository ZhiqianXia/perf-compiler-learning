#include <iostream>
#include <stdexcept>
#include <vector>

struct HermiteNode {
    double x;
    double y;
    double dy;
};

double hermite_interpolate(const std::vector<HermiteNode>& nodes, double x) {
    const std::size_t n = nodes.size();
    std::vector<double> z(2 * n, 0.0);
    std::vector<std::vector<double>> q(2 * n, std::vector<double>(2 * n, 0.0));

    for (std::size_t i = 0; i < n; ++i) {
        z[2 * i] = nodes[i].x;
        z[2 * i + 1] = nodes[i].x;
        q[2 * i][0] = nodes[i].y;
        q[2 * i + 1][0] = nodes[i].y;
        q[2 * i + 1][1] = nodes[i].dy;
        if (i > 0) {
            q[2 * i][1] = (q[2 * i][0] - q[2 * i - 1][0]) / (z[2 * i] - z[2 * i - 1]);
        }
    }

    for (std::size_t j = 2; j < 2 * n; ++j) {
        for (std::size_t i = j; i < 2 * n; ++i) {
            q[i][j] = (q[i][j - 1] - q[i - 1][j - 1]) / (z[i] - z[i - j]);
        }
    }

    double result = q[0][0];
    double term = 1.0;
    for (std::size_t i = 1; i < 2 * n; ++i) {
        term *= (x - z[i - 1]);
        result += q[i][i] * term;
    }
    return result;
}

int main() {
    std::vector<HermiteNode> nodes{{0.0, 1.0, 1.0}, {1.0, 2.718281828, 2.718281828}};
    std::cout << hermite_interpolate(nodes, 0.5) << '\n';
    return 0;
}
