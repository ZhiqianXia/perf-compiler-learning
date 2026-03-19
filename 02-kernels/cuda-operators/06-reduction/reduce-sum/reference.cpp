#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

float reduce_sum_reference(const std::vector<float>& input) {
    return std::accumulate(input.begin(), input.end(), 0.0f);
}

int main() {
    std::vector<float> input(1024);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>((i % 17) - 8);
    }

    std::cout << "n=" << input.size() << "\n";
    std::cout << "sum=" << reduce_sum_reference(input) << "\n";
    return 0;
}
