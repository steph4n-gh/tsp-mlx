import mlx.core as mx
import mlx.nn as nn
import types
from .sparse_cache import SparsePositionTracker

def patch_rope_for_sparse_positions(model: nn.Module, tracker: SparsePositionTracker):
    """
    Monkey-patches the RoPE layers to use our SparsePositionTracker instead of MLX's
    default contiguous `offset + arange` logic. Automatically detects traditional/neox style and freq scaling.
    """
    from .inference import find_layers
    layers = find_layers(model)
    if layers is None:
        print("[TSP] Warning: Could not find transformer layers for RoPE patching.")
        return

    for layer in layers:
        attention = None
        for attr in ["self_attn", "attn", "attention"]:
            if hasattr(layer, attr):
                attention = getattr(layer, attr)
                break
                
        if attention is not None and hasattr(attention, 'rope'):
            rope_layer = attention.rope
            rope_cls = rope_layer.__class__
            
            class PatchedRoPE(rope_cls):
                def __call__(self, x, offset):
                    seq_len = x.shape[2]
                    
                    true_positions = tracker.get_positions()[-seq_len:]
                    positions = mx.array(true_positions, dtype=x.dtype)
                    
                    scale = getattr(self, "scale", 1.0)
                    scaled_positions = positions.astype(mx.float32) * scale
                    
                    if hasattr(self, "_freqs"):
                        freqs_div = self._freqs
                        theta = scaled_positions[:, None] / freqs_div[None, :]
                    elif hasattr(self, "base"):
                        half_dims = self.dims // 2
                        freqs = mx.exp(-mx.arange(0, half_dims, dtype=mx.float32) * (mx.log(self.base) / half_dims))
                        theta = scaled_positions[:, None] * freqs[None, :]
                    else:
                        raise ValueError("RoPE layer must have base or _freqs")
                    
                    costheta = mx.cos(theta)
                    sintheta = mx.sin(theta)
                    
                    for _ in range(x.ndim - 2):
                        costheta = mx.expand_dims(costheta, 0)
                        sintheta = mx.expand_dims(sintheta, 0)
                        
                    x_rope = x[..., :self.dims]
                    x_pass = x[..., self.dims:]

                    if getattr(self, "traditional", False):
                        x1 = x_rope[..., 0::2]
                        x2 = x_rope[..., 1::2]
                        
                        rx1 = x1 * costheta - x2 * sintheta
                        rx2 = x1 * sintheta + x2 * costheta
                        
                        rx = mx.concatenate([rx1[..., None], rx2[..., None]], axis=-1)
                        rx = rx.reshape(x_rope.shape)
                    else:
                        half_dims = self.dims // 2
                        x1 = x_rope[..., :half_dims]
                        x2 = x_rope[..., half_dims:]
                        
                        rx1 = x1 * costheta - x2 * sintheta
                        rx2 = x1 * sintheta + x2 * costheta
                        
                        rx = mx.concatenate([rx1, rx2], axis=-1)
                        
                    if x_pass.shape[-1] > 0:
                        return mx.concatenate([rx, x_pass], axis=-1)
                    return rx
                    
            rope_layer.__class__ = PatchedRoPE
