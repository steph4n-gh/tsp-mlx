#include "compression.hpp"
#include <cmath>
#include <iostream>
#include <filesystem>

namespace tsp {

SubManifoldAutoencoder::SubManifoldAutoencoder(int hidden_dim, int compression_ratio) 
    : hidden_dim_(hidden_dim),
      proj_in_weight_(mlx::core::array(0.0f)),
      proj_in_bias_(mlx::core::array(0.0f)),
      proj_out_weight_(mlx::core::array(0.0f)),
      proj_out_bias_(mlx::core::array(0.0f)) {
    
    int bottleneck_dim = hidden_dim / compression_ratio;
    if (bottleneck_dim == 0) bottleneck_dim = 1;

    // Try to load pre-trained distillation weights
    std::string weight_path = "../../assets/autoencoder_weights.safetensors";
    
    if (std::filesystem::exists(weight_path)) {
        std::cout << "[TSP] \U0001F5DC Loading pre-trained Context Distillation weights..." << std::endl;
        auto st_load = mlx::core::load_safetensors(weight_path);
        auto weights = st_load.first;
        
        // We assume we are loading the keys autoencoder for this unified scaffold
        if (weights.find("k_proj_in.weight") != weights.end()) {
            proj_in_weight_ = mlx::core::transpose(weights.at("k_proj_in.weight"));
            proj_in_bias_ = weights.at("k_proj_in.bias");
            proj_out_weight_ = mlx::core::transpose(weights.at("k_proj_out.weight"));
            proj_out_bias_ = weights.at("k_proj_out.bias");
            
            // Evaluate to realize
            mlx::core::eval({proj_in_weight_, proj_in_bias_, proj_out_weight_, proj_out_bias_});
            std::cout << "[TSP] \U0001F5DC Autoencoder weights loaded successfully!" << std::endl;
            return;
        } else {
             std::cerr << "[TSP] Warning: Pre-trained weights file missing keys. Falling back to random init." << std::endl;
        }
    }
    
    std::cout << "[TSP] \U0001F5DC No pre-trained weights found. Initializing random Autoencoder..." << std::endl;
    // Fan-in / Fan-out initialization
    float scale_in = std::sqrt(2.0f / hidden_dim);
    proj_in_weight_ = mlx::core::random::uniform(-scale_in, scale_in, {hidden_dim, bottleneck_dim});
    proj_in_bias_ = mlx::core::zeros({bottleneck_dim});

    float scale_out = std::sqrt(2.0f / bottleneck_dim);
    proj_out_weight_ = mlx::core::random::uniform(-scale_out, scale_out, {bottleneck_dim, hidden_dim});
    proj_out_bias_ = mlx::core::zeros({hidden_dim});
}

mlx::core::array SubManifoldAutoencoder::operator()(const mlx::core::array& island_tensors) {
    // island_tensors: [1, n_kv_heads, seq_len, head_dim]
    
    // 1. Mean-pooling across the sequence length (axis 2)
    auto macro_vector = mlx::core::mean(island_tensors, std::vector<int>{2}, /*keepdims=*/true); 
    // Now: [1, n_kv_heads, 1, head_dim]
    
    // 2. Pass through bottleneck
    // Linear 1: x @ W + b
    auto compressed = mlx::core::matmul(macro_vector, proj_in_weight_) + proj_in_bias_;
    
    // ReLU
    compressed = mlx::core::maximum(compressed, mlx::core::array(0.0f));
    
    // Linear 2
    auto macro_token = mlx::core::matmul(compressed, proj_out_weight_) + proj_out_bias_;
    
    return macro_token;
}

} // namespace tsp
