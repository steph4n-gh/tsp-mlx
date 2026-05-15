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
        self.inherited_sinks = set()
        self.inheritance_threshold = 0.8
        
        self.enable_compression = enable_compression
        self.enable_consolidation = enable_consolidation
        self.topological_pages = {}
        self.untrusted_indices = set()
        
        if self.enable_compression:
            self.compressor_k, self.compressor_v = load_pretrained_autoencoders(head_dim)
        if self.enable_consolidation:
            self.consolidator = MemoryConsolidator(model=model)

    def update(self, attention_matrix: mx.array, kv_caches: List[Tuple[mx.array, mx.array]], sinks: List[int], x: mx.array = None) -> List[Tuple[mx.array, mx.array]]:
        action = "ALLOW"
        island_indices = []
        combined_sinks = sinks
        
        if attention_matrix is not None:
            # --- Fuzzy Sinks Logic ---
            a_2d = mx.mean(attention_matrix, axis=1)[0]
            if len(a_2d.shape) == 2 and a_2d.shape[0] == 1:
                a_sq = mx.squeeze(a_2d, axis=0) # Shape: [S] for decode
            else:
                a_sq = a_2d # Shape: [S, S] for prefill
                
            if len(a_sq.shape) == 1:
                # Avoid full numpy conversion for all elements. 
                # Only transfer indices that meet the threshold.
                mask = a_sq > self.inheritance_threshold
                if mx.any(mask):
                    import numpy as np
                    high_attn_indices = np.argwhere(np.array(mask)).tolist()
                    for rel_idx_list in high_attn_indices:
                        rel_idx = rel_idx_list[0]
                        if rel_idx < len(self.position_tracker.position_ids):
                            abs_idx = self.position_tracker.position_ids[rel_idx]
                            if abs_idx not in sinks:
                                self.inherited_sinks.add(abs_idx)
            
            combined_sinks = list(set(sinks) | self.inherited_sinks)
    
            decision = self.cortex_hook.evaluate_attention(attention_matrix, combined_sinks, self.position_tracker.position_ids)
            action = decision.get("action", "ALLOW")
            island_indices = decision.get("island_indices", [])
        
        # --- HARD CAP ENFORCEMENT ---
        budget = getattr(self.cortex_hook, "max_context_budget", 4096)
        if len(self.position_tracker.position_ids) > budget and (action != "GARBAGE_COLLECT" or len(island_indices) == 0):
            action = "GARBAGE_COLLECT"
            seq_len = len(self.position_tracker.position_ids)
            immune_window_size = min(50, seq_len)
            immune_set = set(self.position_tracker.position_ids[-immune_window_size:])
            sink_set = set(combined_sinks)
            
            island_indices = []
            excess = len(self.position_tracker.position_ids) - budget + 50 # Prune enough to get safely under budget
            for pid in self.position_tracker.position_ids:
                if pid not in sink_set and pid not in immune_set:
                    island_indices.append(pid)
                    if len(island_indices) >= excess:
                        break
        # ----------------------------

        if action == "FATAL_BLOCK":
            raise RuntimeError("τ-Spectral Pruner intercepted a Semantic Threat. Halting inference.")

        if action == "GARBAGE_COLLECT" and len(island_indices) > 0:
            seq_len = len(self.position_tracker.position_ids)
            island_set = set(island_indices)
            sink_set = set(combined_sinks)
            
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
            if self.enable_consolidation and x is not None and attention_matrix is not None and x.shape[1] == attention_matrix.shape[-1]:
                # 🛑 FIX: "Read-Only" Sandboxing to prevent AI Trauma
                has_untrusted = any(self.position_tracker.position_ids[i] in self.untrusted_indices for i in island_physical_indices)
                
                if not has_untrusted:
                    salience = self.consolidator.evaluate_salience(attention_matrix, island_physical_indices)
                    if salience >= getattr(self.consolidator, "salience_threshold", 0.5):
                        island_array = mx.array(island_physical_indices, dtype=mx.int32)
                        
                        # Extract the hidden states corresponding to the island tokens
                        # x is [B, L, D]
                        x_island = mx.take(x, island_array, axis=1)
                        
                        # We extract the target values for distillation
                        v_island = mx.take(kv_caches[0].values if hasattr(kv_caches[0], "values") else kv_caches[0][1], island_array, axis=2)
                        
                        self.consolidator.consolidate(x_island, v_island)
                else:
                    import logging
                    logging.getLogger("tsp_engine").info("[TSP] \U0001F6A8 Bypassed TTT Consolidation due to Untrusted (Read-Only) tokens in island.")

            # --- V3 Topological Compression ---
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
                for cache in kv_caches:
                    is_tuple = isinstance(cache, tuple)
                    k = cache[0] if is_tuple else cache.keys
                    v = cache[1] if is_tuple else cache.values
                    
                    k_island = mx.take(k, island_array, axis=2)
                    v_island = mx.take(v, island_array, axis=2)
                    page_data.append((k_island, v_island))
                    
                    k_macro = self.compressor_k(k_island)
                    v_macro = self.compressor_v(v_island)
                    
                    new_k = mx.take(k, keep_array, axis=2)
                    new_v = mx.take(v, keep_array, axis=2)
                    
                    new_macro_physical_idx = keep_indices.index(macro_index)
                    
                    # We need to slice and concatenate in Python MLX since assignment might be restrictive
                    # [before_macro, macro, after_macro]
                    k_before = new_k[:, :, :new_macro_physical_idx, :]
                    k_after  = new_k[:, :, new_macro_physical_idx + 1:, :]
                    final_k  = mx.concatenate([k_before, k_macro, k_after], axis=2)
                    
                    v_before = new_v[:, :, :new_macro_physical_idx, :]
                    v_after  = new_v[:, :, new_macro_physical_idx + 1:, :]
                    final_v  = mx.concatenate([v_before, v_macro, v_after], axis=2)
                    
                    if not is_tuple:
                        # Copy back
                        # Cache objects in mlx_lm have specific properties
                        if hasattr(cache, "max_size"):
                            cache.keys[:, :, :final_k.shape[2], :] = final_k
                            cache.values[:, :, :final_v.shape[2], :] = final_v
                        else:
                            cache.keys = final_k
                            cache.values = final_v
                        cache.offset = final_k.shape[2]
                        pruned_caches.append(cache)
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
                for k_p, v_p in page_data:
                    eval_targets.extend([k_p, v_p])
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
                    
                    new_k = mx.take(k, keep_array, axis=2)
                    new_v = mx.take(v, keep_array, axis=2)
                    
                    if not is_tuple:
                        if hasattr(cache, "max_size"):
                            cache.keys[:, :, :new_k.shape[2], :] = new_k
                            cache.values[:, :, :new_v.shape[2], :] = new_v
                        else:
                            cache.keys = new_k
                            cache.values = new_v
                        cache.offset = new_k.shape[2]
                        pruned_caches.append(cache)
                    else:
                        pruned_caches.append((new_k, new_v))
                
                self.position_tracker.prune(list(island_set))
            
            # Prune edges
            self.cortex_hook.edges = {e for e in self.cortex_hook.edges if e[0] not in island_set and e[1] not in island_set}
            
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
            
            k_page, v_page = tensors[i]
            
            k_before = k[:, :, :macro_physical_idx, :]
            k_after  = k[:, :, macro_physical_idx + 1:, :]
            final_k  = mx.concatenate([k_before, k_page, k_after], axis=2)
            
            v_before = v[:, :, :macro_physical_idx, :]
            v_after  = v[:, :, macro_physical_idx + 1:, :]
            final_v  = mx.concatenate([v_before, v_page, v_after], axis=2)
            
            if not is_tuple:
                if hasattr(cache, "max_size"):
                    cache.keys[:, :, :final_k.shape[2], :] = final_k
                    cache.values[:, :, :final_v.shape[2], :] = final_v
                else:
                    cache.keys = final_k
                    cache.values = final_v
                cache.offset = final_k.shape[2]
                unpacked_caches.append(cache)
            else:
                unpacked_caches.append((final_k, final_v))
                
        del self.topological_pages[macro_pos_id]
        print(f"\n[TSP] \U0001F4E6 Topological Compression Triggered: Unpacked {len(pos_ids)} tokens back into active cache!")
        return unpacked_caches
