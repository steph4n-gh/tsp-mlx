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
            
            if "PatchedRoPE" in rope_cls.__name__:
                continue # Already patched, prevent infinite recursion
            
            class PatchedRoPE(rope_cls):
                # Save a reference to the original, un-patched call method
                orig_call = rope_cls.__call__

                def __call__(self, x, offset=0, **kwargs):
                    seq_len = x.shape[2] # 🛑 FIX: The sequence length is at index 2
                    
                    if seq_len > 1:
                        # 🛑 THE PREFILL FIX: If cache is pruned, native cache.offset is wrong.
                        if hasattr(model, "_tsp_kv_manager") and model._tsp_kv_manager is not None:
                            true_offset = model._tsp_kv_manager.position_tracker.current_pos - seq_len
                        else:
                            true_offset = offset
                        return PatchedRoPE.orig_call(self, x, offset=true_offset, **kwargs)

                    # --- DECODE PHASE (L == 1) MANUAL SPARSE MATH ---
                    if hasattr(model, "_tsp_kv_manager") and model._tsp_kv_manager is not None:
                        current_tracker = model._tsp_kv_manager.position_tracker
                    else:
                        current_tracker = tracker
                        
                    all_positions = current_tracker.get_positions()
                    
                    if all_positions.shape[0] < seq_len:
                        positions = mx.arange(offset, offset + seq_len, dtype=x.dtype)
                    else:
                        true_positions = all_positions[-seq_len:]
                        # Avoid allocating a new array buffer. `true_positions` is already an mx.array.
                        positions = true_positions.astype(x.dtype)
                    
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
                    
                    # 🛑 CRITICAL FIX: CAST BACK TO HIDDEN STATE DTYPE
                    costheta = mx.cos(theta).astype(x.dtype)
                    sintheta = mx.sin(theta).astype(x.dtype)
                    
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
