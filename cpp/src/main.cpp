#include "sparse_cache.hpp"
#include "attention_hook.hpp"
#include <iostream>
#include <mlx/mlx.h>

int main() {
    std::cout << "[TSP] Initializing C++ Spectral Pruner with Topological Compression..." << std::endl;
    
    // Enable compression mode and specify head_dim (64)
    tsp::KVCacheManager manager(0.015, true, 64);
    
    // Mock data for testing
    // 10 tokens, each attending to the past
    int seq_len = 10;
    manager.position_tracker().step(seq_len);
    
    // Mock attention matrix [1, 1, 10, 10]
    std::vector<float> buffer(seq_len * seq_len, 0.0f);
    // Add some "islands": tokens 0-4 attend to each other, 5-9 attend to each other
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            buffer[i * seq_len + j] = 0.2f;
        }
    }
    for (int i = 5; i < 10; ++i) {
        for (int j = 5; j < 10; ++j) {
            buffer[i * seq_len + j] = 0.2f;
        }
    }
    
    // Add one link to make it a connected graph but with a clear bisection
    buffer[4 * seq_len + 5] = 0.01f;
    buffer[5 * seq_len + 4] = 0.01f; // Symmetrize for the mock
    
    auto attn = mlx::core::array(buffer.data(), {1, 1, seq_len, seq_len}, mlx::core::float32);

    // Mock KV caches
    auto k = mlx::core::random::uniform(0.0f, 1.0f, {1, 1, seq_len, 64});
    auto v = mlx::core::random::uniform(0.0f, 1.0f, {1, 1, seq_len, 64});
    std::vector<std::pair<mlx::core::array, mlx::core::array>> caches = {{k, v}};

    std::cout << "[TSP] Running Mock Update..." << std::endl;
    auto pruned_caches = manager.update(attn, caches, {0, 1}); // Token 0 and 1 are sinks

    std::cout << "[TSP] Update complete." << std::endl;
    if (pruned_caches[0].first.shape(2) < seq_len) {
        std::cout << "[TSP] SUCCESS: Tokens were pruned. New seq_len: " << pruned_caches[0].first.shape(2) << std::endl;
    } else {
        std::cout << "[TSP] No tokens were pruned in this step." << std::endl;
    }

    return 0;
}
