import os
import json
import mlx.core as mx
from typing import List, Tuple, Any

def save_tsp_state(
    save_dir: str,
    kv_caches: List[Any],
    position_ids: List[int],
    edges: set,
    inherited_sinks: set = None
):
    """
    Serializes the highly-compressed TSP KV cache and topological map to disk.
    """
    if inherited_sinks is None:
        inherited_sinks = set()
        
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Save Topological Map (Position IDs and Edges)
    map_path = os.path.join(save_dir, "topology.json")
    topology = {
        "position_ids": position_ids,
        "edges": list(edges), # Convert set back to list for JSON
        "inherited_sinks": list(inherited_sinks)
    }
    with open(map_path, "w") as f:
        json.dump(topology, f)
        
    # 2. Save KV Cache Tensors
    cache_path = os.path.join(save_dir, "kv_cache.safetensors")
    tensor_dict = {}
    for i, cache in enumerate(kv_caches):
        if hasattr(cache, "keys") and hasattr(cache, "values"):
            # Some mlx_lm cache implementations pad the tensors and use `offset`
            offset = getattr(cache, "offset", cache.keys.shape[2])
            tensor_dict[f"layer_{i}_k"] = cache.keys[:, :, :offset, :]
            tensor_dict[f"layer_{i}_v"] = cache.values[:, :, :offset, :]
        elif isinstance(cache, tuple) and len(cache) == 2:
            # Raw tuples
            tensor_dict[f"layer_{i}_k"] = cache[0]
            tensor_dict[f"layer_{i}_v"] = cache[1]
            
    mx.save_safetensors(cache_path, tensor_dict)
    print(f"[TSP] State successfully checkpointed to {save_dir}")

def load_tsp_state(
    load_dir: str,
    kv_caches: List[Any],
    kv_manager: Any
) -> bool:
    """
    Restores the TSP KV cache and topological map from disk into live VRAM.
    Modifies kv_caches and kv_manager in-place.
    Returns True if successful.
    """
    map_path = os.path.join(load_dir, "topology.json")
    cache_path = os.path.join(load_dir, "kv_cache.safetensors")
    
    if not os.path.exists(map_path) or not os.path.exists(cache_path):
        return False
        
    # 1. Restore Topological Map
    with open(map_path, "r") as f:
        topology = json.load(f)
        
    kv_manager.position_tracker.position_ids = topology["position_ids"]
    kv_manager.position_tracker.current_pos = max(topology["position_ids"]) + 1 if topology["position_ids"] else 0
    kv_manager.cortex_hook.edges = set(tuple(e) for e in topology.get("edges", []))
    kv_manager.inherited_sinks = set(topology.get("inherited_sinks", []))
    
    # 2. Restore KV Cache Tensors
    tensor_dict = mx.load(cache_path)
    
    for i, cache in enumerate(kv_caches):
        k_key = f"layer_{i}_k"
        v_key = f"layer_{i}_v"
        
        if k_key in tensor_dict and v_key in tensor_dict:
            k_tensor = tensor_dict[k_key]
            v_tensor = tensor_dict[v_key]
            
            if hasattr(cache, "keys") and hasattr(cache, "values"):
                # Check if we need to pad based on max_size
                if hasattr(cache, "max_size") and cache.keys.shape[2] > k_tensor.shape[2]:
                    # We are loading into a pre-allocated static cache
                    cache.keys[:, :, :k_tensor.shape[2], :] = k_tensor
                    cache.values[:, :, :v_tensor.shape[2], :] = v_tensor
                else:
                    cache.keys = k_tensor
                    cache.values = v_tensor
                    
                cache.offset = k_tensor.shape[2]
            elif isinstance(cache, tuple) and len(cache) == 2:
                # We can't easily mutate tuples in-place if they are just passed,
                # This assumes the caller will handle the raw tuple return if needed,
                # but usually mlx_lm returns objects.
                pass

    print(f"[TSP] State successfully restored from {load_dir}")
    return True
