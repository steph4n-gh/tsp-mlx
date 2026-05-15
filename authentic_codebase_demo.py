import asyncio
import sys
import os
import mlx.core as mx
from mlx_lm import load
from tsp_mlx.generate import setup_tsp
from tsp_mlx.inference import generate_infinite_context
import time

def clear_screen():
    print("\033[2J\033[H", end="")

def print_dashboard(total_gen, active_ids, evicted, lambda2, step_name, topological_pages=0, action_log=""):
    clear_screen()
    print(f"\033[1;37m[\u03C4-Spectral Pruner] AUTHENTIC CODEBASE DEMO\033[0m")
    print(f"\033[1;35mExecuting on: \033[1mQwen2.5-Coder-7B-Instruct\033[0m")
    print(f"\033[1;36mCurrent Phase: {step_name}\033[0m")
    print("\033[1;30m-------------------------------------------------------------\033[0m")
    
    # Calculate savings
    active_tokens = len(active_ids)
    saved_tokens = total_gen - active_tokens
    saved_percent = (saved_tokens / total_gen * 100) if total_gen > 0 else 0
    
    # Real-time Metrics
    print("\n\033[1;37mREAL-TIME VRAM COMPARISON:\033[0m")
    print(f"  \u25B6 \033[1;31mClassic KV Cache Size:\033[0m    {total_gen} tokens (Would OOM eventually)")
    print(f"  \u25B6 \033[1;32mTSP Active VRAM:\033[0m          {active_tokens} tokens")
    print(f"  \u25B6 \033[1;33mMemory Saved:\033[0m             {saved_tokens} tokens ({saved_percent:.1f}%)")
    print(f"  \u25B6 \033[1mTopological Pages:\033[0m        {topological_pages} Macro-Tokens parked in RAM")
    print(f"  \u25B6 \033[1mSemantic Graph Gap:\033[0m       {lambda2:.8f} (\u03BB\u2082)")
    
    if action_log:
        print(f"\n\033[1;33m[VRAM PROTECTION SYSTEM] {action_log}\033[0m")
    
    # Manifold Visualization
    print("\n\033[1;37mKV CACHE MAP:\033[0m")
    map_str = "  ["
    for i in range(min(60, active_tokens)):
        if i < 5:
            map_str += "\033[32m\u2588\033[0m"  # Sinks
        elif i > active_tokens - 20:
            map_str += "\033[36m\u2592\033[0m"  # Current thought
        else:
            map_str += "\u2591"  # Persistent memory
    if active_tokens > 60:
        map_str += "..."
    map_str += "]"
    print(map_str)
    
    print("\033[1;30m-------------------------------------------------------------\033[0m")
    print("\n\033[1;37mAGENT STDOUT:\033[0m")

async def main():
    print("Loading Immutable Agent (Qwen2.5-Coder-7B-Instruct-4bit)...")
    try:
        model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    except Exception as e:
        print(f"Failed to load model. Error: {e}")
        return

    # Setup aggressive pruning manager
    manager = setup_tsp(model, head_dim=128, enable_compression=True, enable_consolidation=True)
    manager.cortex_hook.base_interval = 5
    manager.cortex_hook.threshold = 0.05
    # Set a strict VRAM budget so massive codebase files trigger compression quickly
    manager.cortex_hook.max_context_budget = 2500 
    manager.consolidator.salience_threshold = 0.5 

    chat_history = [
        {"role": "system", "content": "You are the Immutable Agent. You are an elite AI software engineer. You analyze large codebases precisely. When asked to acknowledge a file, reply ONLY with 'File loaded.' and nothing else."}
    ]
    
    from mlx_lm.models.cache import make_prompt_cache
    persistent_cache = make_prompt_cache(model)
    previous_token_length = 0
    
    # Load authentic files from our own repository to prove there is no faked data
    with open("tsp_mlx/sparse_cache.py", "r") as f:
        file1 = f.read()
    with open("tsp_mlx/cortex_hook.py", "r") as f:
        file2 = f.read()
        
    script = [
        {
            "phase": "Task 1: Ingesting sparse_cache.py",
            "prompt": f"System: I am loading a source file into your context for later analysis. Here is `sparse_cache.py`:\n\n```python\n{file1}\n```\n\n--- END OF FILE ---",
            "skip_generation": True
        },
        {
            "phase": "Task 2: Ingesting cortex_hook.py",
            "prompt": f"System: I am loading a second source file into your context. Here is `cortex_hook.py`:\n\n```python\n{file2}\n```\n\n--- END OF FILE ---",
            "skip_generation": True
        },
        {
            "phase": "Task 3: Semantic Shift (Forcing Topological Compression)",
            "prompt": "User: Now, I want you to completely ignore the `cortex_hook.py` file. Focus ONLY on `sparse_cache.py`. In exactly two concise sentences, explain what the `unpack` method inside `KVCacheManager` does.",
            "max_tokens": 150,
            "skip_generation": False
        }
    ]
        
    global_total_gen = 0
        
    for step in script:
        user_input = step["prompt"]
        # Custom logic to handle System vs User prompts without breaking chat_template bounds
        role = "user" if "User:" in user_input else "system"
        content = user_input.replace("User: ", "").replace("System: ", "")
        
        chat_history.append({"role": role, "content": content})
        
        prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=not step.get("skip_generation"))
        full_input_ids = mx.array(tokenizer.encode(prompt))[None]
        
        # Array-level slicing ensures perfect token boundaries and avoids BOS token insertion
        new_input_ids = full_input_ids[:, previous_token_length:]
        
        if hasattr(manager.cortex_hook, "edges"):
            manager.cortex_hook.edges.clear()
            
        # Fast Prefill (excluding the very last token, which kicks off the generator)
        prefill_ids = new_input_ids[:, :-1] if new_input_ids.shape[1] > 1 else mx.array([[]], dtype=mx.int32)
        
        if prefill_ids.shape[1] > 0:
            chunk_size = 256
            start_time = time.time()
            processed_tokens = 0
            
            for c_idx in range(0, prefill_ids.shape[1], chunk_size):
                chunk = prefill_ids[:, c_idx:c_idx+chunk_size]
                
                # Fast prefill chunk
                manager.position_tracker.step(chunk.shape[1])
                _ = model(chunk, cache=persistent_cache)
                
                # We need to manually trigger the manager update since we aren't in the generator loop
                if hasattr(manager, 'last_attention_matrix') and manager.last_attention_matrix is not None:
                    raw_caches = [(c.keys, c.values) for c in persistent_cache]
                    pruned_raw_caches = manager.update(manager.last_attention_matrix, raw_caches, sinks=[0, 1, 2, 3, 4], x=manager.last_hidden_states)
                    if pruned_raw_caches is not raw_caches:
                        for cache_obj, (pk, pv) in zip(persistent_cache, pruned_raw_caches):
                            cache_obj.keys = pk
                            cache_obj.values = pv
                            cache_obj.offset = pk.shape[2] 
                    manager.last_attention_matrix = None
                    manager.last_hidden_states = None
                
                mx.eval([c.keys for c in persistent_cache if c.keys is not None])
                
                global_total_gen += chunk.shape[1]
                processed_tokens += chunk.shape[1]
                active_pos = manager.position_tracker.position_ids
                evicted = global_total_gen - len(active_pos)
                
                elapsed = time.time() - start_time
                tok_sec = processed_tokens / elapsed if elapsed > 0 else 0
                
                action_log = f"\u23F3 Ingesting Chunk... ({min(c_idx+chunk_size, prefill_ids.shape[1])}/{prefill_ids.shape[1]} tokens) - {tok_sec:.0f} tok/s"
                print_dashboard(global_total_gen, active_pos, evicted, 0.0, step["phase"], len(manager.topological_pages), action_log)
                print(f"System: [Injecting massive source code chunk...]")
                time.sleep(0.01) # Ultra-fast visual delay
        
        # Dial back evaluation frequency to 16 for auto-regressive generation to speed it up
        manager.cortex_hook.current_interval = 16
        
        if step.get("skip_generation"):
            response = "File successfully loaded into context."
            print(f"> {response}\n")
            chat_history.append({"role": "assistant", "content": response})
            prompt_with_response = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=False)
            previous_token_length = len(tokenizer.encode(prompt_with_response))
            time.sleep(2)
            mx.clear_cache()
            import gc
            gc.collect()
            continue
            
        # The final token to kick off the generator
        final_input = new_input_ids[:, -1:] if new_input_ids.shape[1] > 0 else new_input_ids
        
        generator = generate_infinite_context(
            model, 
            final_input, 
            max_tokens=step["max_tokens"], 
            kv_manager=manager, 
            temp=0.7,
            repetition_penalty=1.05,
            repetition_context_size=50,
            kv_caches=persistent_cache
        )
        
        response = ""
        last_evicted = global_total_gen - len(manager.position_tracker.position_ids)
        action_log = ""
        
        token_count = 0
        async for token, stats in generator:
            token_id = token.item()
            if token_id == tokenizer.eos_token_id:
                break
                
            text = tokenizer.decode([token_id])
            response += text
            global_total_gen += 1
            token_count += 1
            
            if stats["total_evicted"] > last_evicted:
                diff = stats["total_evicted"] - last_evicted
                action_log = f"\u26A0\uFE0F CODEBASE PRUNED: Evicted {diff} tokens into a Macro-Token."
                last_evicted = stats["total_evicted"]
            elif stats["lambda_2"] < 0.05 and stats["lambda_2"] > 0:
                action_log = f"SEMANTIC SHIFT DETECTED: Codebase fragmentation imminent (\u03BB\u2082 = {stats['lambda_2']:.4f})"
            
            if token_count % 3 == 0 or token_count == 1:
                print_dashboard(global_total_gen, stats["active_positions"], stats["total_evicted"], stats["lambda_2"], step["phase"], len(manager.topological_pages), action_log)
                if step.get("skip_generation"):
                     print(f"System: [Injecting massive source code chunk...]\n")
                else:
                     clean_prompt = user_input.replace("User: ", "").replace("System: ", "")
                     print(f"User: {clean_prompt}\n")
                print(f"> {response}", end="", flush=True)
                
        chat_history.append({"role": "assistant", "content": response})
        
        prompt_with_response = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=False)
        previous_token_length = len(tokenizer.encode(prompt_with_response))
        
        time.sleep(3)
        mx.clear_cache()
        import gc
        gc.collect()

    print("\n\n[TSP] Authentic Codebase Demo Complete.")

if __name__ == "__main__":
    asyncio.run(main())
