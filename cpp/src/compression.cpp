#include "compression.hpp"
#include <cmath>

namespace tsp {

SubManifoldAutoencoder::SubManifoldAutoencoder(int hidden_dim, int compression_ratio) 
    : hidden_dim_(hidden_dim),
      proj_in_weight_(mlx::core::array(0.0f)),
      proj_in_bias_(mlx::core::array(0.0f)),
      proj_out_weight_(mlx::core::array(0.0f)),
      proj_out_bias_(mlx::core::array(0.0f)) {
    
    int bottleneck_dim = hidden_dim / compression_ratio;
    if (bottleneck_dim == 0) bottleneck_dim = 1;

    // Initialize simple projection weights
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
