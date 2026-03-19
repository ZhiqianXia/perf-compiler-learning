#include <cmath>
#include <iostream>
#include <random>
#include <vector>

int main() {
    std::mt19937 generator(2026);
    std::normal_distribution<double> distribution(0.0, 1.0);
    std::vector<double> samples(10000);
    for (double& sample : samples) {
        sample = distribution(generator);
    }

    double mean = 0.0;
    for (double sample : samples) {
        mean += sample / static_cast<double>(samples.size());
    }
    double variance = 0.0;
    for (double sample : samples) {
        variance += (sample - mean) * (sample - mean) / static_cast<double>(samples.size());
    }

    std::cout << "mean = " << mean << '\n';
    std::cout << "stddev = " << std::sqrt(variance) << '\n';
    return 0;
}
