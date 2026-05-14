#include "attention_hook.hpp"
#include <cmath>

namespace tsp {

mlx::core::array process_attention(
    mlx::core::array queries,
    mlx::core::array keys,
    std::optional<mlx::core::array> mask
) {
    // queries: [B, H, Lq, D]
    // keys: [B, H, Lk, D]
    float scale = 1.0f / std::sqrt(static_cast<float>(queries.shape(-1)));
    
    // Q * K^T
    auto scores = mlx::core::matmul(queries * scale, mlx::core::transpose(keys, {0, 1, 3, 2}));
    
    if (mask.has_value()) {
        scores = scores + mask.value();
    }
    
    auto weights = mlx::core::softmax(scores, -1);
    return weights;
}

} // namespace tsp
