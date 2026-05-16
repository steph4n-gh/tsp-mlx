import mlx.core as mx
from typing import List, Tuple
from .compression import load_pretrained_autoencoders
from .consolidation import MemoryConsolidator

class SparsePositionTracker:
    def __init__(self):
        self._positions = mx.array([], dtype=mx.int32)
        self._position_list = [] # 🛑 FIX: Fast Python lookup cache
        self.current_pos: int = 0

    @property
    def position_ids(self) -> List[int]:
        # Instant O(1) Python list return. Zero GPU syncs.
        return self._position_list

    @position_ids.setter
    def position_ids(self, value: List[int]):
        self._position_list = value
        self._positions = mx.array(value, dtype=mx.int32)

    def step(self, num_tokens: int = 1):
        # Update the Python list
        new_pos_list = list(range(self.current_pos, self.current_pos + num_tokens))
        self._position_list.extend(new_pos_list)
        
        # Update the MLX array
        new_pos = mx.arange(self.current_pos, self.current_pos + num_tokens, dtype=mx.int32)
        self._positions = mx.concatenate([self._positions, new_pos])
        
        self.current_pos += num_tokens

    def prune(self, island_indices: List[int], compressed_index: int = None):
        if not island_indices:
            return
            
        # 🛑 FIX: Blazing fast Python set math. 
        # completely eliminates the massive MLX boolean graph loop.
        island_set = set(island_indices)
        if compressed_index is not None:
            island_set.discard(compressed_index) # Re-enable if part of island
            
        self._position_list = [p for p in self._position_list if p not in island_set]
        
        # Single, clean push to VRAM
        self._positions = mx.array(self._position_list, dtype=mx.int32)

    def get_positions(self) -> mx.array:
        return self._positions

class KVCacheManager:
    def __init__(self, cortex_hook, model: nn.Module = None, enable_compression: bool = True, enable_consolidation: bool = True, head_dim: int = 64):
        self.cortex_hook = cortex_hook
        self.position_tracker = SparsePositionTracker()
        
        self.enable_compression = enable_compression
        self.enable_consolidation = enable_consolidation
        self.topological_pages = {}
        self.untrusted_indices = set()
        
        if self.enable_compression:
            self.compressor_k, self.compressor_v = load_pretrained_autoencoders(head_dim)
        if self.enable_consolidation:
            self.consolidator = MemoryConsolidator(model=model)

    def update(self, attention_matrix: mx.array, kv_caches: List[Tuple], sinks: List[int]) -> List[Tuple]:
        action = "ALLOW"
        island_indices = []
        combined_sinks = sinks

        if attention_matrix is not None:
            decision = self.cortex_hook.evaluate_attention(attention_matrix, combined_sinks, self.position_tracker.position_ids)
            action = decision.get("action", "ALLOW")
            island_indices = decision.get("island_indices", [])
        
        # --- HARD CAP ENFORCEMENT ---
        budget = getattr(self.cortex_hook, "max_context_budget", 8192)
        if len(self.position_tracker.position_ids) > budget:
            action = "GARBAGE_COLLECT"
            seq_len = len(self.position_tracker.position_ids)
            immune_window_size = min(50, seq_len)
            immune_set = set(self.position_tracker.position_ids[-immune_window_size:])
            
            # 🛑 FIX: Macro-Token Protection
            # Macro-Tokens are dense semantic anchors that represent hundreds of compressed tokens.
            # If we don't protect them, the engine will select them for eviction (since they are old),
            # compress them AGAIN, and permanently overwrite the original topological page in RAM!
            # We strictly add them to the sink_set so they are armor-plated in VRAM.
            macro_token_ids = set(self.topological_pages.keys())
            sink_set = set(combined_sinks) | macro_token_ids
            
            # 🛑 FIX: Enforced Deep Clean (Anti-Thrashing)
            # If the Spectral Bisection algorithm found a natural "Thought Island" that is sufficiently 
            # large, we prioritize dropping that exact mathematical cluster to preserve semantic purity.
            # However, if the natural island is smaller than 500 tokens, we must pad it with the oldest 
            # available tokens to enforce a massive 500-token flush. This prevents the active context 
            # from staying permanently glued to the budget limit (which maximizes O(N^2) GPU overhead).
            target_eviction = len(self.position_tracker.position_ids) - budget + 500 
            target_eviction = min(target_eviction, 500) # Max 500 to prevent Macro-Token semantic collapse
            
            if len(island_indices) < target_eviction:
                existing_island = set(island_indices)
                for pid in self.position_tracker.position_ids:
                    if pid not in sink_set and pid not in immune_set and pid not in existing_island:
                        island_indices.append(pid)
                        existing_island.add(pid)
                        if len(island_indices) >= target_eviction:
                            break
        # ----------------------------

        if action == "FATAL_BLOCK":
            raise RuntimeError("τ-Spectral Pruner intercepted a Semantic Threat. Halting inference.")

        if action == "GARBAGE_COLLECT" and len(island_indices) > 0:
            seq_len = len(self.position_tracker.position_ids)
            island_set = set(island_indices)
            sink_set = set(combined_sinks) | set(self.topological_pages.keys())
            
            # IMMUNE WINDOW: Protect the most recent 50 tokens (the active sentence/thought)
            immune_window_size = min(50, seq_len)
            immune_set = set(self.position_tracker.position_ids[-immune_window_size:])
            
            # ARMOR: Never drop a sink or a token in the immune window
            island_set = island_set - sink_set - immune_set
            
            island_physical_indices = [
                i for i in range(seq_len) 
                if self.position_tracker.position_ids[i] in island_set
            ]
            
            if len(island_physical_indices) == 0:
                return kv_caches
                
            keep_indices = [
                i for i in range(seq_len) 
                if self.position_tracker.position_ids[i] not in island_set
            ]
            
            # --- V4 Memory Consolidation (Test-Time Training) ---
            if self.enable_consolidation and attention_matrix is not None:
                # 🛑 FIX: "Read-Only" Sandboxing to prevent AI Trauma
                has_untrusted = any(self.position_tracker.position_ids[i] in self.untrusted_indices for i in island_physical_indices)
                
                if not has_untrusted:
                    salience = self.consolidator.evaluate_salience(attention_matrix, island_physical_indices)
                    if salience >= getattr(self.consolidator, "salience_threshold", 0.5):
                        island_array = mx.array(island_physical_indices, dtype=mx.int32)
                        
                        # Extract the target values for distillation from the LAST layer
                        last_layer_cache = kv_caches[-1]
                        v_layer = last_layer_cache[1] if isinstance(last_layer_cache, tuple) else last_layer_cache.values
                        v_island = mx.take(v_layer, island_array, axis=2)
                        
                        # Extract the saved hidden states from the LAST layer
                        px = last_layer_cache[2] if isinstance(last_layer_cache, tuple) and len(last_layer_cache) >= 3 else getattr(last_layer_cache, 'x_states', None)
                        if px is not None:
                            x_island = mx.take(px, island_array, axis=1)
                            self.consolidator.consolidate(x_island, v_island)
                        else:
                            import logging
                            logging.getLogger("tsp_engine").info("[TSP] \U0001F6A8 Warning: TTT skipped. x_states not found in cache.")
                else:
                    import logging
                    logging.getLogger("tsp_engine").info("[TSP] \U0001F6A8 Bypassed TTT Consolidation due to Untrusted (Read-Only) tokens in island.")

            # --- V3 Topological Compression ---
            macro_pos_id = None
            if self.enable_compression and len(island_physical_indices) > 1:
                # Compression disabled print to keep UI clean
                macro_index = island_physical_indices[0]
                keep_indices.append(macro_index)
                keep_indices.sort()
                macro_pos_id = self.position_tracker.position_ids[macro_index]
                
                island_array = mx.array(island_physical_indices, dtype=mx.int32)
                keep_array = mx.array(keep_indices, dtype=mx.int32)
                
                pruned_caches = []
                page_data = []
                for i in range(len(kv_caches)):
                    cache = kv_caches[i]
                    is_tuple = isinstance(cache, tuple)
                    k = cache[0] if is_tuple else cache.keys
                    v = cache[1] if is_tuple else cache.values
                    px = cache[2] if is_tuple and len(cache) >= 3 else getattr(cache, 'x_states', None)
                    
                    k_island = mx.take(k, island_array, axis=2)
                    v_island = mx.take(v, island_array, axis=2)
                    
                    new_k = mx.contiguous(mx.take(k, keep_array, axis=2))
                    new_v = mx.contiguous(mx.take(v, keep_array, axis=2))
                    
                    new_macro_physical_idx = keep_indices.index(macro_index)
                    
                    if px is not None:
                        px_island = mx.take(px, island_array, axis=1)
                        # 🛑 FIX: TRUE TOPOLOGICAL COMPRESSION
                        # Instead of throwing away 499 tokens with a slice, we use the VarianceCompressor
                        # to statistically extract the principal semantic component with stochastic wiggle.
                        if not hasattr(self, "compressor_px"):
                            from .compression import VarianceCompressor
                            self.compressor_px = VarianceCompressor(hidden_dim=px.shape[-1], tau_wiggle=0.05)
                        
                        px_macro = self.compressor_px(px_island)
                        
                        new_px = mx.contiguous(mx.take(px, keep_array, axis=1))
                        px_before = new_px[:, :new_macro_physical_idx, :]
                        px_after  = new_px[:, new_macro_physical_idx + 1:, :]
                        final_px = mx.contiguous(mx.concatenate([px_before, px_macro, px_after], axis=1))
                        page_data.append((k_island, v_island, px_island))
                    else:
                        final_px = None
                        page_data.append((k_island, v_island))
                        
                    # Since k and v are already RoPE rotated, averaging them directly causes phase cancellation.
                    # If we don't have the original unrotated `px` to re-project, we fallback to slicing the first token
                    # to serve as a positional anchor, while relying on the TTT LoRA updates to handle the actual semantic memory.
                    k_macro = k_island[:, :, 0:1, :]
                    v_macro = v_island[:, :, 0:1, :]
                    
                    k_before = new_k[:, :, :new_macro_physical_idx, :]
                    k_after  = new_k[:, :, new_macro_physical_idx + 1:, :]
                    final_k  = mx.contiguous(mx.concatenate([k_before, k_macro, k_after], axis=2))
                    
                    v_before = new_v[:, :, :new_macro_physical_idx, :]
                    v_after  = new_v[:, :, new_macro_physical_idx + 1:, :]
                    final_v  = mx.contiguous(mx.concatenate([v_before, v_macro, v_after], axis=2))
                    
                    if not is_tuple:
                        # Copy back
                        # Cache objects in mlx_lm have specific properties
                        if hasattr(cache, "max_size"):
                            cache.keys[:, :, :final_k.shape[2], :] = final_k
                            cache.values[:, :, :final_v.shape[2], :] = final_v
                            if final_px is not None:
                                cache.x_states[:, :final_px.shape[1], :] = final_px
                        else:
                            cache.keys = final_k
                            cache.values = final_v
                            if final_px is not None:
                                cache.x_states = final_px
                        cache.offset = final_k.shape[2]
                        pruned_caches.append(cache)
                    else:
                        if len(cache) >= 3:
                            pruned_caches.append((final_k, final_v, final_px))
                        else:
                            pruned_caches.append((final_k, final_v))
                        
                # Store the Topological Page in background RAM
                original_island_pos_ids = [self.position_tracker.position_ids[i] for i in island_physical_indices]
                self.topological_pages[macro_pos_id] = {
                    "pos_ids": original_island_pos_ids,
                    "tensors": page_data
                }
                
                # 🛑 CRITICAL FIX: Evaluate the page tensors immediately!
                # If we don't eval them, they hold a reference to the ENTIRE original KV cache
                # tensors (via the 'take' op), leading to a massive memory leak.
                eval_targets = []
                for p_tuple in page_data:
                    for p_tensor in p_tuple:
                        eval_targets.append(p_tensor)
                mx.eval(*eval_targets)
                
                self.position_tracker.prune(list(island_set), compressed_index=macro_pos_id)
            else:
                # EVICTION MODE (Classic TSP)
                keep_array = mx.array(keep_indices, dtype=mx.int32)
                pruned_caches = []
                
                for cache in kv_caches:
                    is_tuple = isinstance(cache, tuple)
                    k = cache[0] if is_tuple else cache.keys
                    v = cache[1] if is_tuple else cache.values
                    px = cache[2] if is_tuple and len(cache) >= 3 else getattr(cache, 'x_states', None)
                    
                    new_k = mx.contiguous(mx.take(k, keep_array, axis=2))
                    new_v = mx.contiguous(mx.take(v, keep_array, axis=2))
                    new_px = mx.contiguous(mx.take(px, keep_array, axis=1)) if px is not None else None
                    
                    if not is_tuple:
                        if hasattr(cache, "max_size"):
                            cache.keys[:, :, :new_k.shape[2], :] = new_k
                            cache.values[:, :, :new_v.shape[2], :] = new_v
                            if new_px is not None:
                                cache.x_states[:, :new_px.shape[1], :] = new_px
                        else:
                            cache.keys = new_k
                            cache.values = new_v
                            if new_px is not None:
                                cache.x_states = new_px
                        cache.offset = new_k.shape[2]
                        pruned_caches.append(cache)
                    if is_tuple:
                        if len(cache) >= 3:
                            pruned_caches.append((new_k, new_v, new_px))
                        else:
                            pruned_caches.append((new_k, new_v))
                
                self.position_tracker.prune(list(island_set))
            
            # Remap edges to the macro token instead of deleting them to prevent graph fragmentation
            macro_id = macro_pos_id if (self.enable_compression and len(island_physical_indices) > 1) else None
            
            new_edges = set()
            for u, v in self.cortex_hook.edges:
                if u in island_set and v in island_set:
                    continue # Internal island edge, drop
                elif u in island_set:
                    if macro_id is not None:
                        new_edges.add((macro_id, v))
                elif v in island_set:
                    if macro_id is not None:
                        new_edges.add((u, macro_id))
                else:
                    new_edges.add((u, v))
                    
            if macro_id is not None and len(self.position_tracker.position_ids) > 0:
                # 🛑 CRITICAL FIX: Anchor the Macro-Token to the System Prompt (Sink 0)
                # If an island is completely disconnected, the Macro-Token will also be disconnected,
                # causing lambda_2 to permanently drop to 0.0 and triggering endless anomalies.
                # Anchoring it artificially preserves graph continuity.
                system_anchor = self.position_tracker.position_ids[0]
                new_edges.add((macro_id, system_anchor))
                new_edges.add((system_anchor, macro_id))
                    
            self.cortex_hook.edges = new_edges
            
            # 🛑 CRITICAL FIX: Explicitly evaluate the reshaped KV Caches
            # Since MLX uses lazy evaluation, `mx.take` operations remain as pending graph nodes.
            # If we don't evaluate them here, the attention mechanism will re-run the massive prune graph
            # on every single subsequent token generation, causing speed to plummet to < 3 tok/s.
            eval_targets_prune = []
            for cache in pruned_caches:
                if isinstance(cache, tuple):
                    eval_targets_prune.extend([t for t in cache if t is not None])
                else:
                    if hasattr(cache, "keys") and cache.keys is not None: eval_targets_prune.append(cache.keys)
                    if hasattr(cache, "values") and cache.values is not None: eval_targets_prune.append(cache.values)
                    if hasattr(cache, "x_states") and cache.x_states is not None: eval_targets_prune.append(cache.x_states)
            mx.eval(*eval_targets_prune)
            
            return pruned_caches

        return kv_caches

    def unpack(self, macro_pos_id: int, kv_caches: List[Tuple[mx.array, mx.array]]):
        if macro_pos_id not in self.topological_pages:
            return kv_caches
            
        page = self.topological_pages[macro_pos_id]
        pos_ids = page["pos_ids"]
        tensors = page["tensors"]
        
        current_pos_ids = self.position_tracker.position_ids
        try:
            macro_physical_idx = current_pos_ids.index(macro_pos_id)
        except ValueError:
            return kv_caches
            
        # Rebuild position tracker
        new_pos_ids = current_pos_ids[:macro_physical_idx] + pos_ids + current_pos_ids[macro_physical_idx+1:]
        self.position_tracker.position_ids = new_pos_ids
        
        # Rebuild caches
        unpacked_caches = []
        for i, cache in enumerate(kv_caches):
            is_tuple = isinstance(cache, tuple)
            k = cache[0] if is_tuple else cache.keys
            v = cache[1] if is_tuple else cache.values
            px = cache[2] if is_tuple and len(cache) >= 3 else getattr(cache, 'x_states', None)
            
            page_tuple = tensors[i]
            k_page = page_tuple[0]
            v_page = page_tuple[1]
            px_page = page_tuple[2] if len(page_tuple) >= 3 else None
            
            k_before = k[:, :, :macro_physical_idx, :]
            k_after  = k[:, :, macro_physical_idx + 1:, :]
            final_k  = mx.contiguous(mx.concatenate([k_before, k_page, k_after], axis=2))
            
            v_before = v[:, :, :macro_physical_idx, :]
            v_after  = v[:, :, macro_physical_idx + 1:, :]
            final_v  = mx.contiguous(mx.concatenate([v_before, v_page, v_after], axis=2))
            
            if px is not None and px_page is not None:
                px_before = px[:, :macro_physical_idx, :]
                px_after  = px[:, macro_physical_idx + 1:, :]
                final_px  = mx.contiguous(mx.concatenate([px_before, px_page, px_after], axis=1))
            else:
                final_px = None
            
            if not is_tuple:
                if hasattr(cache, "max_size"):
                    cache.keys[:, :, :final_k.shape[2], :] = final_k
                    cache.values[:, :, :final_v.shape[2], :] = final_v
                    if final_px is not None:
                        cache.x_states[:, :final_px.shape[1], :] = final_px
                else:
                    cache.keys = final_k
                    cache.values = final_v
                    if final_px is not None:
                        cache.x_states = final_px
                cache.offset = final_k.shape[2]
                unpacked_caches.append(cache)
            else:
                if final_px is not None:
                    unpacked_caches.append((final_k, final_v, final_px))
                else:
                    unpacked_caches.append((final_k, final_v))
                
        del self.topological_pages[macro_pos_id]
        
        # 🛑 CRITICAL FIX: Evaluate the unpacked KV Caches instantly
        eval_targets_unpack = []
        for cache in unpacked_caches:
            if isinstance(cache, tuple):
                eval_targets_unpack.extend([t for t in cache if t is not None])
            else:
                if hasattr(cache, "keys") and cache.keys is not None: eval_targets_unpack.append(cache.keys)
                if hasattr(cache, "values") and cache.values is not None: eval_targets_unpack.append(cache.values)
                if hasattr(cache, "x_states") and cache.x_states is not None: eval_targets_unpack.append(cache.x_states)
        mx.eval(*eval_targets_unpack)
        
        print(f"\n[TSP] \U0001F4E6 Topological Compression Triggered: Unpacked {len(pos_ids)} tokens back into active cache!")
        return unpacked_caches
