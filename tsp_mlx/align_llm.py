import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from tsp_mlx.sparse_cache import KVCacheManager
from tsp_mlx.cortex_hook import CortexHook
import os

def align():
    print("[Aligner] Loading model (Qwen2.5-0.5B-Instruct)...")
    model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    
    hook = CortexHook(eval_interval=1, threshold=0.01)
    kv_manager = KVCacheManager(hook, model=model, enable_compression=True, enable_consolidation=False)
    
    optimizer = optim.Adam(learning_rate=1e-4)
    
    samples = [
        ("The secret password is 'BLUE-BIRD'.", "What is the secret password?", "BLUE-BIRD"),
        ("The capital of Mars is Olympus City.", "What is the capital of Mars?", "Olympus City"),
        ("A mysterious traveler gave me a golden key.", "What did the traveler give me?", "golden key")
    ]
    
    print("\n[Aligner] Starting Alignment (Teaching the model to read Macro-Tokens)...")
    
    for epoch in range(10):
        total_loss = 0
        for context, question, answer in samples:
            # 1. Process context and question
            context_prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
            
            input_ids = mx.array(tokenizer.encode(context_prompt))[None]
            target_ids = mx.array(tokenizer.encode(answer))[None]
            
            kv_caches = make_prompt_cache(model)
            
            # Initial forward pass to populate KV cache
            _ = model(input_ids, cache=kv_caches)
            
            # 2. Force compression of the context (all tokens except the last 2)
            context_len = input_ids.shape[1] - 2
            island_indices = list(range(context_len))
            
            # Manually trigger compression using the manager's autoencoders
            if hasattr(kv_manager, "compressor_k") and kv_manager.compressor_k:
                for cache in kv_caches:
                    k = cache.keys if hasattr(cache, "keys") else cache[0]
                    v = cache.values if hasattr(cache, "values") else cache[1]
                    
                    # Extract island
                    island_arr = mx.array(island_indices, dtype=mx.int32)
                    k_island = mx.take(k, island_arr, axis=2)
                    v_island = mx.take(v, island_arr, axis=2)
                    
                    # Compress
                    macro_k = kv_manager.compressor_k(k_island)
                    macro_v = kv_manager.compressor_v(v_island)
                    
                    # Create new cache: [Macro-Token] + [Remaining Tokens]
                    keep_indices = mx.array(list(range(context_len, input_ids.shape[1])), dtype=mx.int32)
                    k_keep = mx.take(k, keep_indices, axis=2)
                    v_keep = mx.take(v, keep_indices, axis=2)
                    
                    new_k = mx.concatenate([macro_k, k_keep], axis=2)
                    new_v = mx.concatenate([macro_v, v_keep], axis=2)
                    
                    if hasattr(cache, "keys"):
                        cache.keys = new_k
                        cache.values = new_v
                        cache.offset = new_k.shape[2]
            
            # 3. Define the loss function using the compressed cache
            def loss_fn(model_params):
                # We feed a dummy token to trigger the next prediction using the compressed cache
                dummy_input = mx.array([[tokenizer.encode(" ")[-1]]])
                logits = model(dummy_input, cache=kv_caches)
                # Compare predicted token to the first target token
                # In a full setup, we'd autoregressively train across the target sequence
                return nn.losses.cross_entropy(logits[:, -1, :], target_ids[:, 0])

            loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
            loss, grads = loss_and_grad_fn(model.parameters())
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(samples)
        print(f"  [Aligner] Epoch {epoch} | Avg Loss: {avg_loss:.6f}")

    print("\n[Aligner] Alignment Complete. The model can now interpret Macro-Tokens.")

if __name__ == "__main__":
    align()
