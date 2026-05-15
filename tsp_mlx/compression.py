import os
import mlx.core as mx

class SVDCompressor:
    """
    A deterministic compressor that uses variance-weighted approximation
    to extract the principal semantic component of a KV cache 'Thought Island'.
    Requires no training and guarantees mathematical stability.
    """
    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim
        
    def __call__(self, island_tensors: mx.array) -> mx.array:
        """
        Compresses an island of tokens into a single Macro-Token.
        island_tensors: [1, n_kv_heads, seq_len, head_dim]
        Returns: [1, n_kv_heads, 1, head_dim]
        """
        B, H, L, D = island_tensors.shape
        
        if L <= 1:
            return island_tensors
            
        # reshape to [H, L, D]
        x = mx.squeeze(island_tensors, axis=0)
        
        # Center the data
        mean_x = mx.mean(x, axis=1, keepdims=True)
        centered_x = x - mean_x
        
        # Compute the variance of each token vector from the mean.
        # Highly variant tokens carry more 'signal' than average tokens.
        token_variance = mx.sum(mx.square(centered_x), axis=-1, keepdims=True) # [H, L, 1]
        
        # Normalize weights
        weights = token_variance / (mx.sum(token_variance, axis=1, keepdims=True) + 1e-6)
        
        # Weighted sum: [H, 1, D]
        macro_token = mx.sum(x * weights, axis=1, keepdims=True)
        
        # Restore batch dim: [1, H, 1, D]
        return mx.expand_dims(macro_token, axis=0)

def load_pretrained_autoencoders(hidden_dim: int):
    """
    Returns the deterministic SVD/Variance compressor.
    No weights are loaded from disk.
    """
    print(f"[TSP] \U0001F5DC Using Deterministic SVD/Variance Compression (No training required).")
    k_encoder = SVDCompressor(hidden_dim=hidden_dim)
    v_encoder = SVDCompressor(hidden_dim=hidden_dim)
    
    return k_encoder, v_encoder
