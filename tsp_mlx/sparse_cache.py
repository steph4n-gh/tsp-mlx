import mlx.core as mx
from typing import List, Tuple

class SparsePositionTracker:
    def __init__(self):
        self.position_ids: List[int] = []
        self.current_pos: int = 0

    def step(self, num_tokens: int = 1):
        for _ in range(num_tokens):
            self.position_ids.append(self.current_pos)
            self.current_pos += 1

    def prune(self, island_indices: List[int]):
        island_set = set(island_indices)
        self.position_ids = [pid for pid in self.position_ids if pid not in island_set]

    def get_positions(self) -> mx.array:
        return mx.array(self.position_ids, dtype=mx.int32)

class KVCacheManager:
    def __init__(self, cortex_hook):
        self.cortex_hook = cortex_hook
        self.position_tracker = SparsePositionTracker()
        self.inherited_sinks = set()
        self.inheritance_threshold = 0.8

    def update(self, attention_matrix: mx.array, kv_caches: List[Tuple[mx.array, mx.array]], sinks: List[int]) -> List[Tuple[mx.array, mx.array]]:
        # --- Fuzzy Sinks Logic ---
        a_2d = mx.mean(attention_matrix, axis=1)[0]
        if len(a_2d.shape) == 2 and a_2d.shape[0] == 1:
            a_sq = mx.squeeze(a_2d, axis=0) # Shape: [S] for decode
        else:
            a_sq = a_2d # Shape: [S, S] for prefill
            
        if len(a_sq.shape) == 1:
            import numpy as np
            np_a_sq = np.array(a_sq)
            high_attn_indices = np.where(np_a_sq > self.inheritance_threshold)[0].tolist()
            for rel_idx in high_attn_indices:
                if rel_idx < len(self.position_tracker.position_ids):
                    abs_idx = self.position_tracker.position_ids[rel_idx]
                    if abs_idx not in sinks:
                        self.inherited_sinks.add(abs_idx)
        
        combined_sinks = list(set(sinks) | self.inherited_sinks)

        decision = self.cortex_hook.evaluate_attention(attention_matrix, combined_sinks, self.position_tracker.position_ids)
        action = decision.get("action", "ALLOW")
        island_indices = decision.get("island_indices", [])

        if action == "FATAL_BLOCK":
            raise RuntimeError("τ-Spectral Pruner intercepted a Semantic Threat. Halting inference.")

        if action == "GARBAGE_COLLECT" and len(island_indices) > 0:
            seq_len = attention_matrix.shape[-1]
            island_set = set(island_indices)
            sink_set = set(combined_sinks)
            
            # ARMOR: Never drop a sink, even if Rust flags it
            keep_indices = [
                i for i in range(seq_len) 
                if self.position_tracker.position_ids[i] not in island_set 
                or self.position_tracker.position_ids[i] in sink_set
            ]
            
            # Ensure island_set does not contain sinks so they aren't pruned from position_tracker
            island_set = island_set - sink_set
            
            if len(keep_indices) == seq_len:
                return kv_caches

            # print(f"\n[TSP] \U0001F9F9 GARBAGE COLLECT: Evicting {len(island_indices)} tokens from the KV Cache")

            keep_array = mx.array(keep_indices, dtype=mx.int32)
            pruned_caches = []
            
            for k, v in kv_caches:
                new_k = mx.take(k, keep_array, axis=2)
                new_v = mx.take(v, keep_array, axis=2)
                pruned_caches.append((new_k, new_v))
            
            self.position_tracker.prune(list(island_set))
            
            # Prune edges (using set comprehension)
            self.cortex_hook.edges = {e for e in self.cortex_hook.edges if e[0] not in island_set and e[1] not in island_set}
            
            return pruned_caches

        return kv_caches
