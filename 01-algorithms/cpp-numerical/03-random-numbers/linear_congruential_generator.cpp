#include <iomanip>
#include <iostream>
#include <vector>

class LinearCongruentialGenerator {
public:
    explicit LinearCongruentialGenerator(unsigned long long seed) : state_(seed) {}

    double next_uniform() {
        constexpr unsigned long long multiplier = 1664525ULL;
        constexpr unsigned long long increment = 1013904223ULL;
        constexpr unsigned long long modulus = 1ULL << 32;
        state_ = (multiplier * state_ + increment) % modulus;
        return static_cast<double>(state_) / static_cast<double>(modulus);
    }

private:
    unsigned long long state_;
};

int main() {
    LinearCongruentialGenerator generator(123456789ULL);
    for (int i = 0; i < 10; ++i) {
        std::cout << std::fixed << std::setprecision(8) << generator.next_uniform() << '\n';
    }
    return 0;
}
