import sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from tsp_mlx.sparse_cache import KVCacheManager
from tsp_mlx.cortex_hook import CortexHook
from tsp_mlx.inference import patch_rope_for_sparse_positions, patch_attention_for_extraction
import time

def clear_screen():
    print("\033[2J\033[H", end="")

def print_dashboard(total_gen, active_ids, evicted, lambda2, speaker, action_log=""):
    clear_screen()
    print(f"\033[1;37m[\u03C4-Spectral Pruner] FREE-FLOWING INFINITE CHAT DEMO\033[0m")
    print(f"\033[1;36mActive Speaker: \033[1m{speaker}\033[0m")
    print("\033[1;30m-------------------------------------------------------------\033[0m")
    
    # Real-time Metrics
    print("\n\033[1;37mREAL-TIME VRAM METRICS:\033[0m")
    print(f"  \u25B6 \033[1mTotal Tokens Processed:\033[0m {total_gen} tokens")
    print(f"  \u25B6 \033[1mActive Tokens In VRAM:\033[0m  {len(active_ids)} tokens")
    print(f"  \u25B6 \033[1mTokens Evicted:\033[0m         {evicted} tokens saved")
    print(f"  \u25B6 \033[1mSemantic Graph Gap:\033[0m     {lambda2:.8f} (\u03BB\u2082)")
    
    if action_log:
        print(f"\n\033[1;33m[VRAM PROTECTION SYSTEM] {action_log}\033[0m")
    
    # Manifold Visualization
    print("\n\033[1;37mKV CACHE MAP:\033[0m")
    map_str = "  ["
    for i in range(min(60, len(active_ids))):
        if i < 5:
            map_str += "\033[32m\u2588\033[0m"  # Sinks
        elif i > len(active_ids) - 20:
            map_str += "\033[36m\u2592\033[0m"  # Current thought
        else:
            map_str += "\u2591"  # Persistent memory
    if len(active_ids) > 60:
        map_str += "..."
    map_str += "]"
    print(map_str)
    
    print("\033[1;30m-------------------------------------------------------------\033[0m")
    print("\n\033[1;37mLIVE TRANSCRIPT (Infinite Loop):\033[0m")

def main():
    print("Loading Multi-Agent Engine (Qwen2.5-Coder-7B-Instruct)...")
    try:
        model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    except Exception as e:
        print(f"Failed to load model. Error: {e}")
        return

    kv_caches = make_prompt_cache(model)
    _ = model(mx.array([[0]]), cache=kv_caches)
    head_dim = kv_caches[0].keys.shape[-1]

    # 1. Initialize the shared state
    # Use a sane threshold (0.01) so we only drop genuinely disconnected topics, not the active sentence.
    hook = CortexHook(eval_interval=10, threshold=0.01)
    
    # 🛑 CRITICAL: We must disable compression here. 
    # We are using a 7B model, but the autoencoder was trained on a 0.5B model.
    # The script fell back to random initialization. Injecting a random noise Macro-Token
    # into the KV cache instantly breaks the LLM's attention mechanism, causing the '!!!' loop.
    
    # Test-Time Training (Consolidation) is ENABLED. Gradient clipping prevents NaN explosions.
    manager = KVCacheManager(hook, model=model, enable_compression=False, enable_consolidation=True, head_dim=head_dim)
    manager.consolidator.salience_threshold = 0.0 # Force TTT
    model._tsp_kv_manager = manager
    
    patch_rope_for_sparse_positions(model, manager.position_tracker)
    patch_attention_for_extraction(model)
    
    kv_caches = make_prompt_cache(model)
    
    # We'll use a raw string concatenation for the prompt to have full control and prevent re-tokenization bloat.
    system_prompt = "<|im_start|>system\nYou are an AI in a podcast. Keep responses under 3 sentences. Always ask a question.<|im_end|>\n"
    
    seed = "<|im_start|>user\n[Agent Alpha]: I've been thinking about the nature of intelligence. If we have infinite memory, do we eventually run out of original thoughts?<|im_end|>\n"
    
    # Process initial prompt
    initial_ids = mx.array(tokenizer.encode(system_prompt + seed))[None]
    
    manager.position_tracker.step(initial_ids.shape[1])
    _ = model(initial_ids, cache=kv_caches)
    
    total_gen = initial_ids.shape[1]
    last_evicted = 0
    action_log = ""
    
    print_dashboard(total_gen, manager.position_tracker.position_ids, last_evicted, 0.0, "System", "")
    print(f"\033[1;34m[Agent Alpha]: I've been thinking about the nature of intelligence. If we have infinite memory, do we eventually run out of original thoughts?\033[0m\n")
    
    current_speaker = "Agent Beta"
    next_speaker = "Agent Alpha"
    
    # The script maintains the conversation history just for printing, not for the model.
    printed_history = [
        ("\033[1;34m", "[Agent Alpha]: I've been thinking about the nature of intelligence. If we have infinite memory, do we eventually run out of original thoughts?")
    ]
    
    for turn in range(50):
        # Construct the minimal prompt block to trigger the next agent
        turn_prompt = f"<|im_start|>assistant\n[{current_speaker}]: "
        y = mx.array(tokenizer.encode(turn_prompt))[None]
        
        color = "\033[1;32m" if current_speaker == "Agent Beta" else "\033[1;34m"
        response_text = ""
        generated_tokens = []
        
        print_dashboard(total_gen, manager.position_tracker.position_ids, last_evicted, hook.last_lambda_2, current_speaker, action_log)
        for c, text in printed_history[-3:]:
             print(f"{c}{text}\033[0m\n")
             
        print(f"{color}[{current_speaker}]: ", end="", flush=True)
        
        # Generation Loop
        for i in range(100):
            manager.position_tracker.step(y.shape[1])
            logits = model(y, cache=kv_caches)
            
            # Use temperature sampling
            temp = 0.7
            logits_step = logits[:, -1, :] / temp
            token = mx.random.categorical(logits_step, num_samples=1)
            token_id = token.item()
            
            if token_id == tokenizer.eos_token_id or token_id == tokenizer.encode("<|im_end|>")[0]:
                break
                
            y = token # Feed the token back in for the next step
            total_gen += 1
            
            word = tokenizer.decode([token_id])
            response_text += word
            
            # TSP Update
            if hasattr(manager, 'last_attention_matrix'):
                attn_matrix = manager.last_attention_matrix
                hidden_states = getattr(manager, 'last_hidden_states', None)
                raw_caches = [(c.keys, c.values) for c in kv_caches]
                before_len = manager.position_tracker.get_positions().shape[0]
                
                # 🛑 FIX 1: Protect the entire system prompt (first 50 tokens), not just the first 5.
                # If the system prompt is pruned, the model forgets its instructions and degenerates.
                pruned = manager.update(attn_matrix, raw_caches, sinks=list(range(50)), x=hidden_states)
                
                if pruned is not raw_caches:
                    for cache_obj, (pk, pv) in zip(kv_caches, pruned):
                        cache_obj.keys = pk
                        cache_obj.values = pv
                        cache_obj.offset = pk.shape[2]
                
                # 🛑 FIX 2: Properly evaluate ALL arrays to prevent the massive memory leak.
                # mx.eval() does not automatically traverse custom Python objects like KVCache.
                eval_list = [token, manager.position_tracker.get_positions()]
                for c in kv_caches:
                    eval_list.extend([c.keys, c.values])
                mx.eval(*eval_list)
                
                after_len = manager.position_tracker.get_positions().shape[0]
                total_evicted = (before_len - after_len)
                
                if total_evicted > 0:
                    last_evicted += total_evicted
                    action_log = f"\u26A0\uFE0F ORGANIC DRIFT DETECTED: Evicted {total_evicted} tokens of an old tangent."
                    
                    print_dashboard(total_gen, manager.position_tracker.position_ids, last_evicted, hook.last_lambda_2, current_speaker, action_log)
                    for c, text in printed_history[-3:]:
                         print(f"{c}{text}\033[0m\n")
                    print(f"{color}[{current_speaker}]: {response_text}", end="", flush=True)
            
            print(f"{word}", end="", flush=True)
            
        print("\033[0m\n")
        
        # Clean up the response to avoid the agent repeating its own prefix
        clean_response = response_text.replace(f"[{current_speaker}]:", "").strip()
        printed_history.append((color, f"[{current_speaker}]: {clean_response}"))
        
        # Prepare the next agent's input by capping the current turn
        end_turn_prompt = f"<|im_end|>\n<|im_start|>user\n[{current_speaker}]: {clean_response}<|im_end|>\n"
        end_ids = mx.array(tokenizer.encode(end_turn_prompt))[None]
        manager.position_tracker.step(end_ids.shape[1])
        _ = model(end_ids, cache=kv_caches)
        total_gen += end_ids.shape[1]
        
        # Swap speaker
        current_speaker, next_speaker = next_speaker, current_speaker
        time.sleep(2)

    print("\n\n[TSP] Infinite Chat Demo Complete.")

if __name__ == "__main__":
    main()
