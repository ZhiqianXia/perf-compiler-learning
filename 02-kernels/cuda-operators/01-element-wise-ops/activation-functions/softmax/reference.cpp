#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

std::vector<float> softmax_reference(const std::vector<float>& input) {
    const float max_value = *std::max_element(input.begin(), input.end());
    std::vector<float> output(input.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - max_value);
        sum += output[i];
    }
    for (float& value : output) {
        value /= sum;
    }
    return output;
}

}  // namespace

int main() {
    const std::vector<float> input = {1.0f, 2.0f, 3.0f, -4.0f, 6.0f, 0.5f};
    const auto output = softmax_reference(input);
    const float sum = std::accumulate(output.begin(), output.end(), 0.0f);

    std::cout << "input_size=" << input.size() << "\n";
    std::cout << "prob_sum=" << sum << "\n";
    std::cout << "argmax_prob="
              << *std::max_element(output.begin(), output.end()) << "\n";
    return 0;
}
