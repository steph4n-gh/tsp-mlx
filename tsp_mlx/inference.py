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
            # 🛑 FIX: Extract L from x directly.
            # Do absolutely zero math until we verify it is the decode phase.
            B, L, _ = x.shape

            if x is not None and cache is not None:
                offset = getattr(cache, 'offset', 0)
                if hasattr(cache, "max_size"):
                    # Pre-allocate x_states buffer to match KVCache exactly
                    if not hasattr(cache, 'x_states') or cache.x_states is None or cache.x_states.shape[1] < cache.max_size:
                        cache.x_states = mx.zeros((B, cache.max_size, x.shape[2]), dtype=x.dtype)
                    cache.x_states[:, offset:offset+L, :] = x
                else:
                    if not hasattr(cache, 'x_states') or cache.x_states is None:
                        cache.x_states = x
                    else:
                        cache.x_states = mx.concatenate([cache.x_states, x], axis=1)

            if L == 1 and hasattr(self.orig, "q_proj") and hasattr(self.orig, "k_proj") and hasattr(self.orig, "v_proj"):
                if hasattr(model, '_tsp_kv_manager') and model._tsp_kv_manager is not None:
                    # ONLY run the manual projections if we are extracting the matrix
                    queries = self.orig.q_proj(x)
                    keys = self.orig.k_proj(x)
                    values = self.orig.v_proj(x)                    
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
                            k_cache = cache.keys
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
                    
                    if not hasattr(model._tsp_kv_manager, 'layer_scores_accum') or model._tsp_kv_manager.layer_scores_accum is None:
                        model._tsp_kv_manager.layer_scores_accum = scores
                    else:
                        if model._tsp_kv_manager.layer_scores_accum.shape != scores.shape:
                            model._tsp_kv_manager.layer_scores_accum = scores
                        else:
                            model._tsp_kv_manager.layer_scores_accum = mx.maximum(model._tsp_kv_manager.layer_scores_accum, scores)
                            
                    model._tsp_kv_manager.last_scores_matrix = model._tsp_kv_manager.layer_scores_accum
                    
                    del scores # 🛑 FIX: Explicitly drop scores to prevent OOM
                    
                    if not hasattr(model._tsp_kv_manager, 'layer_attn_accum') or model._tsp_kv_manager.layer_attn_accum is None:
                        model._tsp_kv_manager.layer_attn_accum = attn_weights
                    else:
                        if model._tsp_kv_manager.layer_attn_accum.shape != attn_weights.shape:
                            model._tsp_kv_manager.layer_attn_accum = attn_weights
                        else:
                            model._tsp_kv_manager.layer_attn_accum = mx.maximum(model._tsp_kv_manager.layer_attn_accum, attn_weights)
                        
                    model._tsp_kv_manager.last_attention_matrix = model._tsp_kv_manager.layer_attn_accum 
                    
                    # 🛑 FIX: Eliminate Double Computation
                    # Instead of throwing away the math and calling self.orig(), we finish the attention pass.
                    if cache is not None:
                        if hasattr(cache, "values"):
                            v_cache = cache.values
                            if v_cache is not None:
                                offset = getattr(cache, 'offset', v_cache.shape[2])
                                v_cache = v_cache[:, :, :offset, :]
                                full_values = mx.concatenate([v_cache, values], axis=2)
                            else:
                                full_values = values
                        else:
                             full_values = values
                    else:
                        full_values = values

                    if n_heads != n_kv_heads:
                        full_values = mx.repeat(full_values, repeats, axis=1)

                    context = (attn_weights @ full_values).transpose(0, 2, 1, 3).reshape(B, L, -1)
                    if hasattr(self.orig, "o_proj"):
                        return self.orig.o_proj(context)
            
            return self.orig(x, mask=mask, cache=cache, **kwargs)

    setattr(last_layer, attn_attr, AttentionWrapper(orig_attn))


async def generate_infinite_context(
    model: nn.Module, 
    prompt: mx.array, 
    max_tokens: int = 1000,
    kv_manager = None,
    temp: float = 0.0,
    repetition_penalty: float = 1.1,
    repetition_context_size: int = 20,
    kv_caches = None
) -> Generator[Tuple[mx.array, dict], None, None]:
    """
    Yields (token, stats_dict) for instrumentation.
    """
    import asyncio
    
    if kv_caches is None:
        from mlx_lm.models.cache import make_prompt_cache
        kv_caches = make_prompt_cache(model)
        
    head_dim = kv_caches[0].keys.shape[-1] if kv_caches and kv_caches[0].keys is not None else 64

    if kv_manager is None:
        if not hasattr(model, '_tsp_kv_manager'):
            # Use the default library path search logic in CortexHook
            hook = CortexHook(eval_interval=16, threat_threshold=999999.0)
            kv_manager = KVCacheManager(hook, model=model, head_dim=head_dim)
            model._tsp_kv_manager = kv_manager
    
            patch_rope_for_sparse_positions(model, kv_manager.position_tracker)
            patch_attention_for_extraction(model)
        else:
            kv_manager = model._tsp_kv_manager
            
    y = prompt
    total_evicted = 0
    lambda_2 = 0.0
    history_tokens = []
    
    try:
        # --- Pre-Generation Topological Compression Unpack Trigger ---
        # We evaluate the raw attention scores from the prefill phase BEFORE it gets overwritten.
        # Using raw scores bypasses float16 softmax underflow for massive contexts.
        if hasattr(kv_manager, 'last_scores_matrix') and kv_manager.last_scores_matrix is not None and hasattr(kv_manager, 'topological_pages'):
            scores_matrix = kv_manager.last_scores_matrix
            # scores_matrix shape: [B, H, L_new, L_cache]
            scores_mean_heads = mx.mean(scores_matrix, axis=1)[0] # shape: [L_new, L_cache]

            if len(scores_mean_heads.shape) > 1:
                scores_max_seq = mx.max(scores_mean_heads, axis=0) # shape: [L_cache]
            else:
                scores_max_seq = scores_mean_heads

            macro_scores = []
            for macro_pos_id in list(kv_manager.topological_pages.keys()):
                try:
                    physical_idx = kv_manager.position_tracker.position_ids.index(macro_pos_id)
                    if physical_idx < scores_max_seq.shape[0]:
                        score = scores_max_seq[physical_idx].item()
                        macro_scores.append((score, macro_pos_id))
                except ValueError:
                    pass

            # Unpack the top 3 most relevant Macro-Tokens
            macro_scores.sort(reverse=True, key=lambda x: x[0])
            unpack_targets = [m_id for score, m_id in macro_scores[:3]]

            for target in unpack_targets:
                kv_caches = kv_manager.unpack(target, kv_caches)

            kv_manager.layer_attn_accum = None
            kv_manager.layer_scores_accum = None
        # -----------------------------------------

        # We don't compile decode_step here because it drops the AttentionWrapper side effects
        def decode_step(y_step):
            kv_manager.layer_attn_accum = None
            kv_manager.layer_scores_accum = None
            if hasattr(model, "model") and hasattr(model, "lm_head"):
                hidden_states = model.model(y_step, cache=kv_caches)
                logits = model.lm_head(hidden_states)
            else:
                logits = model(y_step, cache=kv_caches)
            return logits, kv_manager.layer_attn_accum

        for i in range(max_tokens):
            kv_manager.position_tracker.step(y.shape[1])
            
            if y.shape[1] == 1:
                logits, attn_matrix = decode_step(y)
                logits = logits[:, -1, :]
                kv_manager.last_attention_matrix = attn_matrix
            else:
                if hasattr(model, "model") and hasattr(model, "lm_head"):
                    hidden_states = model.model(y, cache=kv_caches)
                    logits = model.lm_head(hidden_states[:, -1:, :])
                    logits = logits[:, -1, :]
                else:
                    logits = model(y, cache=kv_caches)
                    logits = logits[:, -1, :]
            
            # --- Apply Repetition Penalty ---
            if repetition_penalty > 1.0 and len(history_tokens) > 0:
                recent_history = list(set(history_tokens[-repetition_context_size:]))
                indices = mx.array(recent_history)
                selected_logits = logits[0, indices]
                selected_logits = mx.where(
                    selected_logits < 0,
                    selected_logits * repetition_penalty,
                    selected_logits / repetition_penalty
                )
                logits[0, indices] = selected_logits
            # --------------------------------
            
            if temp > 0:
                logits_step = logits / temp
                y = mx.random.categorical(logits_step, num_samples=1)
            else:
                y = mx.argmax(logits, axis=-1, keepdims=True)
                
            # 🛑 EXTRACT CACHES
            cache_tensors = [c.keys for c in kv_caches if c.keys is not None] + \
                            [c.values for c in kv_caches if c.values is not None]
            
            if hasattr(kv_manager, 'last_attention_matrix') and kv_manager.last_attention_matrix is not None:
                attn_matrix = kv_manager.last_attention_matrix
                
                # 🛑 CRITICAL FIX: Evaluate EVERYTHING in ONE PASS before hitting Python boundaries
                eval_targets = [y, attn_matrix, kv_manager.position_tracker.get_positions()] + cache_tensors
                mx.eval(*eval_targets)
                
                # NOW safe to cross Python boundary
                history_tokens.append(y.item())
                
                raw_caches = [(c.keys, c.values, getattr(c, 'x_states', None)) for c in kv_caches]
                before_len = kv_manager.position_tracker.get_positions().shape[0]
                
                # (This runs instantly now because attn_matrix is fully realized in Metal)
                pruned_raw_caches = kv_manager.update(attn_matrix, raw_caches, sinks=[0, 1, 2, 3, 4])
                
                after_len = kv_manager.position_tracker.get_positions().shape[0]
                total_evicted += (before_len - after_len)
                
                if hasattr(kv_manager.cortex_hook, 'last_lambda_2'):
                    lambda_2 = kv_manager.cortex_hook.last_lambda_2
                
                if pruned_raw_caches is not raw_caches:
                    for cache_obj, (pk, pv, px) in zip(kv_caches, pruned_raw_caches):
                        cache_obj.keys = pk
                        cache_obj.values = pv
                        cache_obj.offset = pk.shape[2] 
                        if px is not None:
                            cache_obj.x_states = px
                
                kv_manager.last_attention_matrix = None
                kv_manager.last_scores_matrix = None
                
                # Evaluate pruned cache so it's fully realized for the next loop
                new_cache_tensors = [c.keys for c in kv_caches if c.keys is not None] + \
                                    [c.values for c in kv_caches if c.values is not None] + \
                                    [c.x_states for c in kv_caches if hasattr(c, 'x_states') and c.x_states is not None]
                mx.eval(kv_manager.position_tracker.get_positions(), *new_cache_tensors)
                
            else:
                mx.eval(y, kv_manager.position_tracker.get_positions(), *cache_tensors)
                history_tokens.append(y.item())
            
            stats = {
                "active_positions": kv_manager.position_tracker.position_ids.copy(),
                "total_evicted": total_evicted,
                "lambda_2": lambda_2,
                "macro_tokens": len(getattr(kv_manager, 'topological_pages', {}))
            }
            yield y, stats
            
            # 🛑 FIX: Prevent JIT/Buffer Cache OOM during long generations.
            # Dynamic pruning causes unique graph shapes, which MLX caches forever.
            # We must explicitly flush the cache periodically.
            if i > 0 and i % 100 == 0:
                mx.clear_cache()
                
            await asyncio.sleep(0)
    finally:
        # Guarantee rigorous Garbage Collection
        if kv_manager is not None:
            kv_manager.last_attention_matrix = None
            kv_manager.last_hidden_states = None
        mx.clear_cache()

