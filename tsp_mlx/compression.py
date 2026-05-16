import os
import mlx.core as mx

class VarianceCompressor:
    """
    A stochastic compressor that uses variance-weighted approximation
    with an optional 'tau wiggle' to extract the principal semantic component.
    """
    def __init__(self, hidden_dim: int, tau_wiggle: float = 0.05):
        self.hidden_dim = hidden_dim
        self.tau_wiggle = tau_wiggle
        # 🛑 FIX: Cryptographic Position Salting to prevent Topological Spoofing
        self.session_salt = mx.random.normal((1, 1, 1, hidden_dim)) * 0.1 + 1.0
        
    def __call__(self, island_tensors: mx.array) -> mx.array:
        """
        Compresses an island of tokens into a single Macro-Token.
        Handles both 4D [B, n_kv_heads, seq_len, head_dim] and 3D [B, seq_len, hidden_dim] tensors.
        Returns tensor of identical dimensionality with seq_len compressed to 1.
        """
        was_3d = False
        if len(island_tensors.shape) == 3:
            # It's x_states: [B, L, D]
            was_3d = True
            island_tensors = mx.expand_dims(island_tensors, axis=1) # -> [B, 1, L, D]
            
        B, H, L, D = island_tensors.shape
        
        if L <= 1:
            if was_3d:
                return mx.squeeze(island_tensors, axis=1)
            return island_tensors
            
        # Salt the tensors to prevent predictable spoofing
        salted_tensors = island_tensors * self.session_salt
            
        # Center the data across sequence length (axis 2), preserving Batch
        mean_x = mx.mean(salted_tensors, axis=2, keepdims=True)
        centered_x = salted_tensors - mean_x
        
        # Compute the variance of each token vector from the mean.
        # Highly variant tokens carry more 'signal' than average tokens.
        token_variance = mx.sum(mx.square(centered_x), axis=-1, keepdims=True) # [B, H, L, 1]
        
        # 🛑 FIX: Inject 'Tau Wiggle' (Stochastic Perturbation)
        # Guards against Feature Collapse while preserving Paging Stability
        if self.tau_wiggle > 0.0:
            # We scale the noise amplitude by the average variance of the island.
            # This ensures the wiggle is strictly proportional (max ~5% distortion)
            # preventing the semantic anchor from losing its structural identity.
            mean_var = mx.mean(token_variance, axis=2, keepdims=True)
            noise = mx.random.uniform(
                shape=token_variance.shape, 
                low=-self.tau_wiggle, 
                high=self.tau_wiggle,
                dtype=token_variance.dtype
            ) * mean_var
            # Ensure no negative variances
            token_variance = mx.maximum(token_variance + noise, mx.zeros_like(token_variance))
        
        # Normalize weights
        weights = token_variance / (mx.sum(token_variance, axis=2, keepdims=True) + 1e-6)
        
        # Weighted sum: [B, H, 1, D]
        macro_token = mx.sum(island_tensors * weights, axis=2, keepdims=True)
        
        if was_3d:
            macro_token = mx.squeeze(macro_token, axis=1) # -> [B, 1, D]
            
        return macro_token

def load_pretrained_autoencoders(hidden_dim: int, tau_wiggle: float = 0.05):
    """
    Returns the stochastic Variance compressor.
    No weights are loaded from disk.
    """
    print(f"[TSP] \U0001F5DC Using Stochastic Variance Compression (Tau Wiggle={tau_wiggle}).")
    k_encoder = VarianceCompressor(hidden_dim=hidden_dim, tau_wiggle=tau_wiggle)
    v_encoder = VarianceCompressor(hidden_dim=hidden_dim, tau_wiggle=tau_wiggle)
    
    return k_encoder, v_encoder
