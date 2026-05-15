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
        def __init__(self, orig):
            super().__init__()
            self.orig = orig
            for attr in dir(orig):
                if not attr.startswith("__") and not callable(getattr(orig, attr)):
                    try:
                        val = getattr(orig, attr)
                        if not isinstance(val, nn.Module):
                            setattr(self, attr, val)
                    except AttributeError:
                        pass

        def __call__(self, x, mask=None, cache=None, **kwargs):
            if hasattr(self.orig, "q_proj") and hasattr(self.orig, "k_proj") and hasattr(self.orig, "v_proj"):
                queries, keys, values = self.orig.q_proj(x), self.orig.k_proj(x), self.orig.v_proj(x)
                
                B, L, _ = queries.shape
                
                # 🛑 FIX: Prevent OOM by skipping O(N^2) extraction during prefill.
                # Only compute the manual attention matrix during decode (L == 1).
                if L == 1 and hasattr(model, '_tsp_kv_manager') and model._tsp_kv_manager is not None:
                    n_heads = getattr(self.orig, "n_heads", 1)
                    n_kv_heads = getattr(self.orig, "n_kv_heads", n_heads)
                    
                    queries = queries.reshape(B, L, n_heads, -1).transpose(0, 2, 1, 3)
                    keys = keys.reshape(B, L, n_kv_heads, -1).transpose(0, 2, 1, 3)
                    values = values.reshape(B, L, n_kv_heads, -1).transpose(0, 2, 1, 3)

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
                    
                    attn_weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
                    model._tsp_kv_manager.last_attention_matrix = attn_weights 
                    model._tsp_kv_manager.last_hidden_states = x # Capture x for TTT
            
            return self.orig(x, mask=mask, cache=cache, **kwargs)

    setattr(last_layer, attn_attr, AttentionWrapper(orig_attn))


async def generate_infinite_context(
    model: nn.Module, 
    prompt: mx.array, 
    max_tokens: int = 1000,
    kv_manager = None
) -> Generator[Tuple[mx.array, dict], None, None]:
    """
    Yields (token, stats_dict) for instrumentation.
    """
    import asyncio
    if kv_manager is None:
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
            
            # --- Holographic Paging Unpack Trigger ---
            if attn_matrix is not None and y.shape[1] == 1 and hasattr(kv_manager, 'holographic_pages'):
                # attn_matrix shape during decode: [B, H, 1, Lk]
                attn_mean = mx.mean(attn_matrix, axis=1)[0, 0]
                unpack_targets = []
                for macro_pos_id in list(kv_manager.holographic_pages.keys()):
                    try:
                        physical_idx = kv_manager.position_tracker.position_ids.index(macro_pos_id)
                        if physical_idx < attn_mean.shape[0]:
                            # If attention spikes on the macro token, trigger unpack
                            if attn_mean[physical_idx].item() > 0.05:
                                unpack_targets.append(macro_pos_id)
                    except ValueError:
                        pass
                
                for target in unpack_targets:
                    kv_caches = kv_manager.unpack(target, kv_caches)
            # -----------------------------------------
            
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
            
            kv_manager.last_attention_matrix = None
            kv_manager.last_hidden_states = None
            
            cache_tensors = []
            for c in kv_caches:
                if c.keys is not None: cache_tensors.append(c.keys)
                if c.values is not None: cache_tensors.append(c.values)
            
            # 🛑 CRITICAL: Clear the computation graph to prevent memory leaks
            mx.synchronize()
            mx.eval(y, kv_manager.position_tracker.get_positions(), *cache_tensors)
        else:
            cache_tensors = []
            for c in kv_caches:
                if c.keys is not None: cache_tensors.append(c.keys)
                if c.values is not None: cache_tensors.append(c.values)
            mx.synchronize()
            mx.eval(y, *cache_tensors)
        
        stats = {
            "active_positions": kv_manager.position_tracker.position_ids.copy(),
            "total_evicted": total_evicted,
            "lambda_2": lambda_2
        }
        yield y, stats
        
        # 🛑 FIX: Prevent JIT/Buffer Cache OOM during long generations.
        # Dynamic pruning causes unique graph shapes, which MLX caches forever.
        # We must explicitly flush the cache periodically.
        if i > 0 and i % 100 == 0:
            mx.clear_cache()
            
        await asyncio.sleep(0)

