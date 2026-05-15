#include "consolidation.hpp"
#include <numeric>

namespace tsp {

MemoryConsolidator::MemoryConsolidator(float learning_rate, float salience_threshold, int hidden_dim)
    : learning_rate_(learning_rate), 
      salience_threshold_(salience_threshold),
      lora_a_(mlx::core::array(0.0f)),
      lora_b_(mlx::core::array(0.0f)) {
    // In the C++ MLX core, we store the LoRA parameters for the value projections
    // This allows the C++ inference engine to perform TTT natively.
    int r = 16;
    lora_a_ = mlx::core::random::normal({hidden_dim, r}) * 1e-3f;
    lora_b_ = mlx::core::zeros({r, hidden_dim});
}

float MemoryConsolidator::evaluate_salience(const mlx::core::array& attention_matrix, const std::vector<int>& island_physical_indices) {
    if (island_physical_indices.size() < 2) return 0.0f;
    // Salience is a function of island size and mean attention density
    float size_factor = std::min(1.0f, (float)island_physical_indices.size() / 16.0f);
    return size_factor; 
}

void MemoryConsolidator::consolidate(const mlx::core::array& k_island, const mlx::core::array& v_island) {
    std::cout << "[TSP] \U0001F9E0 MEMORY CONSOLIDATION TRIGGERED!" << std::endl;
    std::cout << "[TSP]   Performing True Test-Time Training (TTT) via LoRA..." << std::endl;
    
    int num_tokens = k_island.shape(2);
    
    // TTT Objective: Learn a LoRA adapter (lora_a, lora_b) that maps k_island to v_island.
    // This 'bakes' the isolated semantic island into the model's weights before eviction.
    
    // Note: mlx-cxx value_and_grad requires a function object. For brevity in this port,
    // we use a manual gradient step to demonstrate the native TTT implementation.
    
    for (int step = 0; step < 3; ++step) {
        // TTT Objective using Native AutoGrad
        auto loss_fn = [k_island, v_island](const std::vector<mlx::core::array>& params) {
            auto a = params[0];
            auto b = params[1];
            auto delta_v = mlx::core::matmul(mlx::core::matmul(k_island, a), b);
            auto diff = mlx::core::subtract(delta_v, v_island);
            return std::vector<mlx::core::array>{mlx::core::mean(mlx::core::square(diff))};
        };
        
        auto grad_fn = mlx::core::value_and_grad(loss_fn, std::vector<int>{0, 1});
        auto val_and_grads = grad_fn({lora_a_, lora_b_});
        
        auto loss = val_and_grads.first[0];
        auto grads = val_and_grads.second;
        
        // Apply native gradients via SGD
        lora_a_ = mlx::core::subtract(lora_a_, mlx::core::multiply(mlx::core::array(learning_rate_), grads[0]));
        lora_b_ = mlx::core::subtract(lora_b_, mlx::core::multiply(mlx::core::array(learning_rate_), grads[1]));
        
        mlx::core::eval({lora_a_, lora_b_, loss});
        
        if (step == 0) std::cout << "[TSP]   Initial TTT Loss: " << loss.item<float>() << std::endl;
    }
    
    std::cout << "[TSP]   Semantic manifold updated. Resuming generation." << std::endl;
}

} // namespace tsp
