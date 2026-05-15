import mlx.core as mx
import mlx.nn as nn
from typing import Generator, List, Any, Tuple
import types

from .cortex_hook import CortexHook
from .sparse_cache import KVCacheManager, SparsePositionTracker
from .rope_patches import patch_rope_for_sparse_positions

def find_layers(module, layer_name="layers"):
    if hasattr(module, layer_name):
        return getattr(module, layer_name)
    for name, child in module.named_modules():
        res = find_layers(child, layer_name)
        if res is not None:
            return res
    return None

def patch_attention_for_extraction(model: nn.Module):
    layers = find_layers(model)
    if layers is None or len(layers) == 0:
        print("Warning: Unrecognized model architecture. Cannot patch attention.")
        return

    last_layer = layers[-1]
    
    attn_attr = None
    for attr in ["self_attn", "attn", "attention"]:
        if hasattr(last_layer, attr):
            attn_attr = attr
            break
            
    if attn_attr is None:
         print("Warning: Could not find attention block in the last layer.")
         return

    orig_attn = getattr(last_layer, attn_attr)

    class AttentionWrapper(nn.Module):
        def __init__(self, orig, global_model):
            super().__init__()
            self.orig = orig
            self._global_model = global_model
            for attr in dir(orig):
                if not attr.startswith("__") and not callable(getattr(orig, attr)):
                    try:
                        setattr(self, attr, getattr(orig, attr))
                    except AttributeError:
                        pass

        def __call__(self, x, mask=None, cache=None, **kwargs):
            if hasattr(self.orig, "q_proj") and hasattr(self.orig, "k_proj") and hasattr(self.orig, "v_proj"):
                queries, keys, values = self.orig.q_proj(x), self.orig.k_proj(x), self.orig.v_proj(x)
                
                B, L, _ = queries.shape
                n_heads = getattr(self.orig, "n_heads", 1)
                n_kv_heads = getattr(self.orig, "n_kv_heads", n_heads)
                
                queries = queries.reshape(B, L, n_heads, -1).transpose(0, 2, 1, 3)
                keys = keys.reshape(B, L, n_kv_heads, -1).transpose(0, 2, 1, 3)
                values = values.reshape(B, L, n_kv_heads, -1).transpose(0, 2, 1, 3)

                # 🛑 FIX: APPLY ROPE BEFORE DOT PRODUCT
                if hasattr(self.orig, "rope"):
                    current_offset = cache.offset if cache is not None else 0
                    queries = self.orig.rope(queries, offset=current_offset)
                    keys = self.orig.rope(keys, offset=current_offset)
                
                if cache is not None:
                    if hasattr(cache, "keys"):
                        k_cache, v_cache = cache.keys, cache.values
                        if k_cache is not None:
                            offset = getattr(cache, 'offset', k_cache.shape[2])
                            k_cache = k_cache[:, :, :offset, :]
                            full_keys = mx.concatenate([k_cache, keys], axis=2)
                        else:
                            full_keys = keys
                    else:
                         full_keys = keys 
                else:
                    full_keys = keys

                if n_heads != n_kv_heads:
                    repeats = n_heads // n_kv_heads
                    full_keys = mx.repeat(full_keys, repeats, axis=1)

                scale = 1.0 / mx.sqrt(queries.shape[-1])
                scores = (queries * scale) @ full_keys.transpose(0, 1, 3, 2)
                
                # 🛑 FIX: APPLY CAUSAL MASK AND SOFTMAX
                L_q, L_k = scores.shape[2], scores.shape[3]
                if L_q > 1:  
                    causal_mask = mx.triu(mx.full((L_q, L_k), -float('inf')), k=1)
                    scores = scores + causal_mask

                attn_weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
                if hasattr(self._global_model, '_tsp_kv_manager'):
                    self._global_model._tsp_kv_manager.last_attention_matrix = attn_weights 
                    self._global_model._tsp_kv_manager.last_hidden_states = x # Capture x for TTT
            
            return self.orig(x, mask=mask, cache=cache, **kwargs)

    setattr(last_layer, attn_attr, AttentionWrapper(orig_attn, model))


def generate_infinite_context(
    model: nn.Module, 
    prompt: mx.array, 
    max_tokens: int = 1000
) -> Generator[Tuple[mx.array, dict], None, None]:
    """
    Yields (token, stats_dict) for instrumentation.
    """
    if not hasattr(model, '_tsp_kv_manager'):
        # Use the default library path search logic in CortexHook
        hook = CortexHook(eval_interval=16, threat_threshold=999999.0)
        kv_manager = KVCacheManager(hook, model=model)
        model._tsp_kv_manager = kv_manager

        patch_rope_for_sparse_positions(model, kv_manager.position_tracker)
        patch_attention_for_extraction(model)
    else:
        kv_manager = model._tsp_kv_manager
        kv_manager.position_tracker.position_ids = []
        kv_manager.position_tracker.current_pos = 0
        kv_manager.cortex_hook.edges = set()

    from mlx_lm.models.cache import make_prompt_cache
    kv_caches = make_prompt_cache(model)
    
    y = prompt
    total_evicted = 0
    lambda_2 = 0.0
    
    for i in range(max_tokens):
        kv_manager.position_tracker.step(y.shape[1])
        
        logits = model(y, cache=kv_caches)
        y = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        
        if hasattr(kv_manager, 'last_attention_matrix'):
            attn_matrix = kv_manager.last_attention_matrix
            hidden_states = getattr(kv_manager, 'last_hidden_states', None)
            raw_caches = [(c.keys, c.values) for c in kv_caches]
            before_len = kv_manager.position_tracker.get_positions().shape[0]
            
            # Use sinks = [0,1,2,3,4] to protect system prompt
            pruned_raw_caches = kv_manager.update(attn_matrix, raw_caches, sinks=[0, 1, 2, 3, 4], x=hidden_states)
            
            after_len = kv_manager.position_tracker.get_positions().shape[0]
            total_evicted += (before_len - after_len)
            
            if hasattr(kv_manager.cortex_hook, 'last_lambda_2'):
                lambda_2 = kv_manager.cortex_hook.last_lambda_2
            
            if pruned_raw_caches is not raw_caches:
                for cache_obj, (pk, pv) in zip(kv_caches, pruned_raw_caches):
                    cache_obj.keys = pk
                    cache_obj.values = pv
                    cache_obj.offset = pk.shape[2] 
            
            # 🛑 CRITICAL: Clear the computation graph to prevent memory leaks
            mx.eval(y, kv_caches, kv_manager.position_tracker.get_positions())
        else:
            mx.eval(y, kv_caches)
        
        stats = {
            "active_positions": kv_manager.position_tracker.position_ids.copy(),
            "total_evicted": total_evicted,
            "lambda_2": lambda_2
        }
        yield y, stats

