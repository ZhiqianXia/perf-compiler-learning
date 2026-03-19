#include <cmath>
#include <complex>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <vector>

using Complex = std::complex<double>;

void fft(std::vector<Complex>& values) {
    const std::size_t n = values.size();
    if ((n & (n - 1)) != 0) {
        throw std::invalid_argument("fft input size must be a power of two");
    }

    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(values[i], values[j]);
        }
    }

    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double angle = -2.0 * std::numbers::pi / static_cast<double>(len);
        const Complex wlen(std::cos(angle), std::sin(angle));
        for (std::size_t i = 0; i < n; i += len) {
            Complex w(1.0, 0.0);
            for (std::size_t j = 0; j < len / 2; ++j) {
                const Complex u = values[i + j];
                const Complex v = values[i + j + len / 2] * w;
                values[i + j] = u + v;
                values[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

int main() {
    std::vector<Complex> values{{1.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0},
                                {1.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0}};
    fft(values);
    for (const auto& value : values) {
        std::cout << value << '\n';
    }
    return 0;
}
