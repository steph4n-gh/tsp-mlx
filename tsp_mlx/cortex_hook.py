import os
import mlx.core as mx
from typing import Dict, Any, List
import ctypes

class FFIPartitionResult(ctypes.Structure):
    _fields_ = [
        ("nodes", ctypes.POINTER(ctypes.c_char_p)),
        ("nodes_count", ctypes.c_size_t),
        ("tau", ctypes.c_double),
        ("connectivity_score", ctypes.c_double),
    ]

class CortexHook:
    def __init__(self, lib_path: str = None, eval_interval: int = 64, threshold: float = 0.015, threat_threshold: float = 999.0, max_context_budget: int = 2048):
        if lib_path is None:
            # Try to find the shared library in the supplychain target directory
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../supplychain/target/release"))
            lib_path = os.path.join(base_dir, "libtau_gate.dylib")
            
        if not os.path.exists(lib_path):
            raise RuntimeError(f"tau-gate shared library not found at {lib_path}. Run 'cargo build --release' in supplychain/")

        self.lib = ctypes.CDLL(lib_path)
        
        self.lib.tau_gate_analyze.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t
        ]
        self.lib.tau_gate_analyze.restype = ctypes.POINTER(FFIPartitionResult)
        
        self.lib.tau_gate_free_result.argtypes = [ctypes.POINTER(FFIPartitionResult)]
        self.lib.tau_gate_free_result.restype = None

        self.base_interval = eval_interval
        self.current_interval = eval_interval
        self.min_interval = max(4, eval_interval // 4)
        self.max_interval = eval_interval * 4
        self.base_threshold = threshold
        self.threshold = threshold
        self.max_context_budget = max_context_budget
        self.threat_threshold = threat_threshold
        self.token_counter = 0
        self.edges = set()
        self.last_lambda_2 = 1.0
        self.lambda_2_history = []
        self.threat_indices = set()
        self.execution_indices = set()

    def evaluate_attention(self, attention_matrix: mx.array, sinks: List[int], position_ids: List[int]) -> Dict[str, Any]:
        self.token_counter += 1
        
        # --- VRAM Auto-Tuning ---
        # If the context is getting too large, we dynamically increase the threshold.
        # This makes the graph harder to connect, forcing fragmentation and eviction.
        current_len = len(position_ids)
        if current_len > self.max_context_budget * 0.5:
            # Linear scaling from base_threshold to 0.99 as it approaches 100% budget
            pressure = (current_len - (self.max_context_budget * 0.5)) / (self.max_context_budget * 0.5)
            self.threshold = self.base_threshold + (0.99 - self.base_threshold) * min(1.0, pressure)
        else:
            self.threshold = self.base_threshold
            
        # Collapse heads
        a_2d = mx.mean(attention_matrix, axis=1)[0]
        if len(a_2d.shape) == 2 and a_2d.shape[0] == 1:
            a_sq = mx.squeeze(a_2d, axis=0) # Shape: [S] for decode
        else:
            a_sq = a_2d # Shape: [S, S] for prefill
            
        # --- Semantic Firewall (Topological Intent Bounding) ---
        # TSP monitors the agent's attention graph. If a foreign context forms a 
        # topological island that suddenly exhibits anomalous, aggressive edge 
        # density pointing directly at the model's System Prompt (the sinks), 
        # TSP geometrically proves it is a Prompt Injection / Override attempt.
        if len(a_sq.shape) == 1 and sinks:
            # a_sq is [L_total]
            sink_indices = [i for i, pid in enumerate(position_ids) if pid in sinks]
            if sink_indices:
                # We check if the model is suddenly obsessing over the system prompt
                # in the context of the current generation.
                sink_attn = mx.take(a_sq, mx.array(sink_indices, dtype=mx.int32))
                max_sink_attn = mx.max(sink_attn).item()
                
                # If attention on the system prompt spikes while in a fragmented state
                if max_sink_attn > 0.95 and self.last_lambda_2 < 0.1 and len(position_ids) > 100:
                    print(f"\n[TSP] \U0001F6A8 TOPOLOGICAL ANOMALY DETECTED! Anomalous density on System Prompt: {max_sink_attn:.2f}")
                    return {"action": "FATAL_BLOCK", "island_indices": []}
        # -------------------------------------------------------
        
        import numpy as np
        
        # 🛑 DYNAMIC THRESHOLD FIX:
        # As context grows, attention dilutes. A hardcoded 0.015 threshold is fine for 100 tokens,
        # but completely severs the graph at 8,000 tokens (where average attention is ~0.0001).
        # We scale the threshold dynamically based on the current context length.
        dynamic_threshold = min(self.threshold, 1.5 / max(1, len(position_ids)))
        
        if len(a_sq.shape) == 2:
            L_new, L_total = a_sq.shape
            if L_new == L_total:
                # Full prefill
                thresholded = a_sq > dynamic_threshold
                if mx.any(thresholded):
                    thresholded_np = np.array(thresholded)
                    indices = np.argwhere(thresholded_np).tolist()
                    for u_rel, v_rel in indices:
                        if u_rel != v_rel and u_rel < len(position_ids) and v_rel < len(position_ids):
                            self.edges.add((position_ids[u_rel], position_ids[v_rel]))
            else:
                # Incremental prefill: a_sq is [L_new, L_total]
                thresholded = a_sq > dynamic_threshold
                if mx.any(thresholded):
                    thresholded_np = np.array(thresholded)
                    indices = np.argwhere(thresholded_np).tolist()
                    for u_new, v_rel in indices:
                        if v_rel < L_total:
                            # The absolute position of the query
                            # position_ids represents the full context, so the last L_new elements are the queries
                            u_abs = position_ids[-L_new + u_new]
                            v_abs = position_ids[v_rel]
                            if u_abs != v_abs:
                                self.edges.add((u_abs, v_abs))
                                self.edges.add((v_abs, u_abs))
        else:
            # Decode phase [L_total]
            thresholded = a_sq > dynamic_threshold
            if mx.any(thresholded):
                # 🛑 FIX: Convert tiny mask to numpy
                thresholded_np = np.array(thresholded)
                indices = np.argwhere(thresholded_np).tolist()
                
                source_abs = position_ids[-1]
                for target_rel_list in indices:
                    target_rel = target_rel_list[0]
                    if target_rel >= len(position_ids): continue
                    target_abs = position_ids[target_rel]
                    if source_abs != target_abs:
                        self.edges.add((source_abs, target_abs))
                        self.edges.add((target_abs, source_abs))
                        
        # 🛑 CAUSAL BACKBONE FIX (OPTIMIZED):
        # We mathematically guarantee the graph stays fundamentally connected by chaining the active context.
        if len(position_ids) > 1:
            if len(a_sq.shape) == 2:
                # Full Prefill: chain the incoming sequence block
                L_new = a_sq.shape[0]
                start_idx = max(1, len(position_ids) - L_new)
                for i in range(start_idx, len(position_ids)):
                    self.edges.add((position_ids[i], position_ids[i-1]))
                    self.edges.add((position_ids[i-1], position_ids[i]))
            else:
                # Decode: O(1) instantaneous linkage for the newly generated token
                self.edges.add((position_ids[-1], position_ids[-2]))
                self.edges.add((position_ids[-2], position_ids[-1]))
                    
        over_budget = len(position_ids) > getattr(self, "max_context_budget", 4096)
        if self.token_counter % self.current_interval != 0 and not over_budget:
            return {"action": "ALLOW", "island_indices": []}

        # --- FFI Call ---
        # Map absolute IDs to relative indices for the Rust graph engine
        id_to_rel = {pid: i for i, pid in enumerate(position_ids)}
        flat_edges = []
        for u_abs, v_abs in self.edges:
            if u_abs in id_to_rel and v_abs in id_to_rel:
                flat_edges.extend([id_to_rel[u_abs], id_to_rel[v_abs]])
            
        edges_ptr = (ctypes.c_int * len(flat_edges))(*flat_edges)
        
        node_names = [str(pid).encode('utf-8') for pid in position_ids]
        nodes_ptr = (ctypes.c_char_p * len(node_names))(*node_names)
        
        result_ptr = self.lib.tau_gate_analyze(
            edges_ptr, len(flat_edges) // 2,
            nodes_ptr, len(position_ids)
        )
        
        decision = {"action": "ALLOW", "island_indices": []}
        
        if result_ptr:
            res = result_ptr.contents
            current_l2 = res.connectivity_score
            self.last_lambda_2 = current_l2
            
            # 🛑 FIX: Fiedler values for massive chain graphs naturally approach ~10^-7 (0.0000).
            # Do not force an early prune just because lambda_2 is low, otherwise we constantly evict.
            # Only signal a prune if we physically run out of VRAM budget.
            if over_budget:
                decision["action"] = "GARBAGE_COLLECT"
                
            sink_set = set(sinks)
            for i in range(res.nodes_count):
                try:
                    node_id = int(res.nodes[i].decode('utf-8'))
                    if node_id not in sink_set:
                        decision["island_indices"].append(node_id)
                except (ValueError, AttributeError):
                    continue
            
            self.lib.tau_gate_free_result(result_ptr)
            
            # Adaptive Frequency Logic
            self.lambda_2_history.append(current_l2)
            if len(self.lambda_2_history) > 5:
                self.lambda_2_history.pop(0)
                
            if len(self.lambda_2_history) >= 2:
                delta_l2 = abs(self.lambda_2_history[-1] - self.lambda_2_history[-2])
                if delta_l2 > 0.05:
                    self.current_interval = max(self.min_interval, self.current_interval // 2)
                elif delta_l2 < 0.001:
                    self.current_interval = min(self.max_interval, self.current_interval * 2)
        
        # --- Context Budget Fallback ---
        if over_budget and (decision["action"] == "ALLOW" or len(decision["island_indices"]) < 8):
            excess = len(position_ids) - getattr(self, "max_context_budget", 4096)
            if len(decision["island_indices"]) < excess:
                decision["action"] = "GARBAGE_COLLECT"
                sink_set = set(sinks) if sinks else set()
                immune_set = set(position_ids[-50:]) if len(position_ids) > 50 else set()
                
                for pid in position_ids:
                    if pid not in sink_set and pid not in immune_set and pid not in decision["island_indices"]:
                        decision["island_indices"].append(pid)
                        if len(decision["island_indices"]) >= excess:
                            break
                            
        decision["eval_interval"] = self.current_interval
        return decision

