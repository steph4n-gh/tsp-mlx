#pragma once
#include <mlx/mlx.h>
#include <vector>
#include <iostream>

namespace tsp {

class MemoryConsolidator {
public:
    MemoryConsolidator(float learning_rate = 1e-4, float salience_threshold = 0.5f);
    
    // Evaluates the internal density (salience) of the island
    float evaluate_salience(const mlx::core::array& attention_matrix, const std::vector<int>& island_physical_indices);
    
    // Simulates a low-rank backward pass to bake knowledge into persistent weights
    void consolidate(const mlx::core::array& k_island, const mlx::core::array& v_island);

private:
    float learning_rate_;
    float salience_threshold_;
};

} // namespace tsp
