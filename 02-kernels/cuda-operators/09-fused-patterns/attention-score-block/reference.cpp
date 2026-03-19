#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

std::vector<float> attention_score_reference(const std::vector<float>& q,
                                             const std::vector<float>& k,
                                             const std::vector<float>& mask,
                                             int batch_size,
                                             int num_heads,
                                             int seq_len,
                                             int head_dim,
                                             float scale) {
    std::vector<float> probs(batch_size * num_heads * seq_len * seq_len, 0.0f);
    std::vector<float> row_scores(seq_len);

    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int row = 0; row < seq_len; ++row) {
                for (int col = 0; col < seq_len; ++col) {
                    float dot = 0.0f;
                    const int q_offset = ((b * num_heads + h) * seq_len + row) * head_dim;
                    const int k_offset = ((b * num_heads + h) * seq_len + col) * head_dim;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += q[q_offset + d] * k[k_offset + d];
                    }
                    row_scores[col] = dot * scale + mask[row * seq_len + col];
                }

                const float row_max = *std::max_element(row_scores.begin(), row_scores.end());
                float row_sum = 0.0f;
                for (float& value : row_scores) {
                    value = std::exp(value - row_max);
                    row_sum += value;
                }
                for (int col = 0; col < seq_len; ++col) {
                    const int index = ((b * num_heads + h) * seq_len + row) * seq_len + col;
                    probs[index] = row_scores[col] / row_sum;
                }
            }
        }
    }

    return probs;
}

int main() {
    constexpr int batch_size = 1;
    constexpr int num_heads = 2;
    constexpr int seq_len = 4;
    constexpr int head_dim = 8;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> q(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> k(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> mask(seq_len * seq_len, 0.0f);
    for (float& value : q) {
        value = dist(rng);
    }
    for (float& value : k) {
        value = dist(rng);
    }
    for (int row = 0; row < seq_len; ++row) {
        for (int col = row + 1; col < seq_len; ++col) {
            mask[row * seq_len + col] = -1e9f;
        }
    }

    const auto probs = attention_score_reference(q, k, mask, batch_size, num_heads, seq_len, head_dim, scale);
    std::cout << "shape=(" << batch_size << "," << num_heads << "," << seq_len << ")\n";
    std::cout << "sample_probs=" << probs[0] << ", " << probs[1] << ", " << probs[2] << "\n";
    return 0;
}
