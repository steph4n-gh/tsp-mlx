#include "sparse_cache.hpp"
#include "attention_hook.hpp"
#include <iostream>
#include <mlx/mlx.h>

int main(int argc, char** argv) {
    std::cout << "[TSP] Initializing Native C++ Spectral Pruner..." << std::endl;
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_tensors.safetensors>" << std::endl;
        std::cerr << "You must provide a real extracted tensor file to prove the native pipeline." << std::endl;
        return 1;
    }
    
    std::string tensor_path = argv[1];
    std::cout << "[TSP] Loading real tensors from: " << tensor_path << std::endl;
    
    auto loaded = mlx::core::load_safetensors(tensor_path);
    auto tensors = loaded.first;
    
    if (tensors.find("attention") == tensors.end() || tensors.find("k") == tensors.end()) {
        std::cerr << "Error: Required tensors ('attention', 'k', 'v') not found in " << tensor_path << std::endl;
        return 1;
    }
    
    auto attn = tensors.at("attention");
    auto k = tensors.at("k");
    auto v = tensors.at("v");
    
    // Enable compression and consolidation modes
    // Using a high threshold (e.g., 0.99) for strict isolation testing if needed, or 0.015 standard
    tsp::KVCacheManager manager(0.015, true, true, k.shape(3));
    
    int seq_len = k.shape(2);
    manager.position_tracker().step(seq_len);
    
    std::vector<std::pair<mlx::core::array, mlx::core::array>> caches = {{k, v}};

    std::cout << "[TSP] Running Native Update on Real Extracted Tensors (seq_len=" << seq_len << ")..." << std::endl;
    
    // Protect the first 5 tokens (system prompt)
    auto pruned_caches = manager.update(attn, caches, {0, 1, 2, 3, 4}); 

    std::cout << "[TSP] Update complete." << std::endl;
    if (pruned_caches[0].first.shape(2) < seq_len) {
        std::cout << "[TSP] \033[1;32mSUCCESS: Tokens were pruned. New seq_len: " << pruned_caches[0].first.shape(2) << "\033[0m" << std::endl;
    } else {
        std::cout << "[TSP] \033[1;33mNo tokens were pruned in this step. Graph is fully connected.\033[0m" << std::endl;
    }

    return 0;
}
