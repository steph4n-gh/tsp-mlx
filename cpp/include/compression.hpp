#pragma once
#include <mlx/mlx.h>

namespace tsp {

class SubManifoldAutoencoder {
public:
    SubManifoldAutoencoder(int hidden_dim, int compression_ratio = 4);
    
    // Compresses an island of tokens [1, n_kv_heads, seq_len, head_dim]
    // into a single Macro-Token [1, n_kv_heads, 1, head_dim]
    mlx::core::array operator()(const mlx::core::array& island_tensors);

private:
    int hidden_dim_;
    // In MLX C++, we can represent simple linear layers manually for now
    // or use mlx::nn if we set up the namespace correctly. For this lightweight
    // autoencoder, we'll maintain two weight matrices.
    mlx::core::array proj_in_weight_;
    mlx::core::array proj_in_bias_;
    
    mlx::core::array proj_out_weight_;
    mlx::core::array proj_out_bias_;
};

} // namespace tsp
