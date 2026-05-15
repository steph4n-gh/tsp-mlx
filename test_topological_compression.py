import mlx.core as mx
from mlx_lm import load
from tsp_mlx.sparse_cache import KVCacheManager
from tsp_mlx.cortex_hook import CortexHook
from tsp_mlx.inference import patch_attention_for_extraction, patch_rope_for_sparse_positions
from mlx_lm.models.cache import make_prompt_cache

model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")

patch_attention_for_extraction(model)

class MockHook:
    def evaluate_attention(self, attn, sinks, pos):
        if len(pos) > 20:
            return {"action": "GARBAGE_COLLECT", "island_indices": pos[10:40]}
        return {"action": "ALLOW", "island_indices": []}
        
hook = MockHook()
hook.edges = set()
hook.last_lambda_2 = 0.5
manager = KVCacheManager(hook, model=model, enable_compression=True, enable_consolidation=False)
model._tsp_kv_manager = manager

# 100 tokens
prompt = mx.array([[0] * 100])
cache = make_prompt_cache(model)

y = prompt
manager.position_tracker.step(y.shape[1])
logits = model(y, cache=cache)
y = mx.array([[1]]) # 1 token

attn = mx.zeros((1, 1, 1, manager.position_tracker.get_positions().shape[0]))
attn[0, 0, 0, 0:5] = 1.0 # Only attend to sinks
manager.last_attention_matrix = attn

print("Before Update, Pos IDs:", len(manager.position_tracker.position_ids))

raw_caches = [(c.keys, c.values) for c in cache]
pruned_caches = manager.update(attn, raw_caches, sinks=[0,1,2,3,4], x=None)
for c, (pk, pv) in zip(cache, pruned_caches):
    c.keys = pk; c.values = pv; c.offset = pk.shape[2]

print("After Update (Compressed), Pos IDs:", len(manager.position_tracker.position_ids))
print("Topological pages:", list(manager.topological_pages.keys()))

# Now simulate an attention spike on the macro token
macro_id = list(manager.topological_pages.keys())[0]
physical_idx = manager.position_tracker.position_ids.index(macro_id)

attn = mx.zeros((1, 1, 1, manager.position_tracker.get_positions().shape[0]))
attn[0, 0, 0, physical_idx] = 1.0 # Spike!
manager.last_attention_matrix = attn

print("\n--- Triggering Unpack! ---")

attn_mean = mx.mean(attn, axis=1)[0, 0]
unpack_targets = []
for m_id in list(manager.topological_pages.keys()):
    p_idx = manager.position_tracker.position_ids.index(m_id)
    if attn_mean[p_idx].item() > 0.05:
        unpack_targets.append(m_id)

for t in unpack_targets:
    cache = manager.unpack(t, cache)
    
print("After Unpack, Pos IDs:", len(manager.position_tracker.position_ids))
print("Topological pages left:", len(manager.topological_pages))
