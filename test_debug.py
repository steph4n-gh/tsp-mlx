import mlx.core as mx
from mlx_lm import load
from tsp_mlx.sparse_cache import KVCacheManager
from tsp_mlx.inference import patch_attention_for_extraction

model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")

class MockHook:
    def evaluate_attention(self, attn, sinks, pos):
        return {"action": "GARBAGE_COLLECT", "island_indices": pos[10:40]}
        
hook = MockHook()
hook.edges = set()
hook.last_lambda_2 = 0.5
manager = KVCacheManager(hook, model=model, enable_compression=True, enable_consolidation=False)

from mlx_lm.models.cache import make_prompt_cache
prompt = mx.array([[0] * 100])
cache = make_prompt_cache(model)
manager.position_tracker.step(prompt.shape[1])
logits = model(prompt, cache=cache)

attn = mx.zeros((1, 1, 1, 100))
attn[0, 0, 0, 0:5] = 1.0 # Only attend to sinks
raw_caches = [(c.keys, c.values) for c in cache]
manager.update(attn, raw_caches, sinks=[0,1,2,3,4], x=None)
print("After update, pos ids:", len(manager.position_tracker.position_ids))
print("Topological pages:", list(manager.topological_pages.keys()))
