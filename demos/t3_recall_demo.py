import asyncio
import sys
import os
import mlx.core as mx
from mlx_lm import load
from tsp_mlx.generate import setup_tsp
from tsp_mlx.inference import generate_infinite_context
import time
import subprocess

def clear_screen():
    print("\033[2J\033[H", end="")

def print_dashboard(total_gen, active_ids, evicted, lambda2, step_name, topological_pages=0, action_log=""):
    clear_screen()
    print(f"\033[1;37m[\u03C4-Spectral Pruner] HIGH-SPEED CODEBASE RECALL DEMO\033[0m")
    print(f"\033[1;35mExecuting on: \033[1mQwen2.5-Coder-7B-Instruct\033[0m")
    print(f"\033[1;36mCurrent Phase: {step_name}\033[0m")
    print("\033[1;30m-------------------------------------------------------------\033[0m")
    
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
    
    # Dynamic Manifold Visualization
    print("\n\033[1;37mDYNAMIC KV CACHE MAP:\033[0m")
    map_str = "  ["
    if total_gen > 0:
        bucket_size = max(1.0, total_gen / 60.0)
        active_set = set(active_ids)
        for i in range(60):
            start = int(i * bucket_size)
            end = int((i + 1) * bucket_size)
            
            # Check if any ID in the active set falls within this bucket
            # We use an intersection approach for performance
            bucket_range = set(range(start, end))
            if bucket_range.intersection(active_set):
                if i < 3:
                    map_str += "\033[1;32m\u2588\033[0m"  # System Sinks (Green)
                elif i > 56:
                    map_str += "\033[1;36m\u2592\033[0m"  # Working Memory (Cyan)
                else:
                    map_str += "\033[1;37m\u2588\033[0m"  # Standard Active Memory (White)
            else:
                map_str += "\033[1;31m\u00B7\033[0m"     # Evicted Topology (Red Dot)
    else:
        map_str += "\u2591" * 60
        
    map_str += "]"
    print(map_str)
    
    print("\033[1;30m-------------------------------------------------------------\033[0m")
    print("\n\033[1;37mAGENT STDOUT:\033[0m")

async def main():
    print("Loading Immutable Agent (Qwen2.5-Coder-7B-Instruct-4bit)...")
    try:
        model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-8bit")
    except Exception as e:
        print(f"Failed to load model. Error: {e}")
        return

    # 🛑 FORCE VRAM WARM-UP
    print("Warming up Unified Memory...")
    mx.eval(model.parameters())
    mx.synchronize()

    # Setup aggressive pruning manager with a tight budget
    manager = setup_tsp(model, head_dim=128, enable_compression=True, enable_consolidation=True)
    manager.cortex_hook.base_interval = 5
    manager.cortex_hook.threshold = 0.05
    manager.cortex_hook.max_context_budget = 2000 
    manager.consolidator.salience_threshold = 0.0 # Force TTT on all evictions to ensure semantic alignment

    chat_history = [
        {"role": "system", "content": "You are the Immutable Agent. You read massive codebases quickly. You must be extremely concise when answering questions about the codebase."}
    ]
    
    from mlx_lm.models.cache import make_prompt_cache
    persistent_cache = make_prompt_cache(model)
    previous_string_length = 0
    
    repo_url = "https://github.com/pingdotgg/t3code.git"
    clone_dir = "/tmp/t3code-demo"
    
    if not os.path.exists(clone_dir):
        print(f"[\u03C4-Spectral Pruner] Downloading authentic pingdotgg/t3code repository from GitHub...")
        subprocess.run(["git", "clone", "--depth", "1", repo_url, clone_dir], capture_output=True, check=True)
        
    base_path = clone_dir
    
    files_to_load = [
        "apps/server/src/provider/Layers/ClaudeAdapter.ts",
        "apps/web/src/components/ChatView.tsx"
    ]
    file_contents = {}
    
    for f_path in files_to_load:
        path = os.path.join(base_path, f_path)
        if os.path.exists(path):
            with open(path, "r") as f:
                file_contents[f_path] = f.read()
        else:
            print(f"Error: Could not find {path}")
            return
            
    script = []
    
    for i, f_path in enumerate(files_to_load):
        f_name = os.path.basename(f_path)
        script.append({
            "phase": f"Task {i+1}: Ingesting {f_name}",
            "prompt": f"System: I am loading a massive source file into your context. Here is `{f_name}`:\n\n```tsx\n{file_contents[f_path]}\n```\n\n--- END OF FILE ---",
            "skip_generation": True,
            "filename": f_name
        })
        
    script.append({
        "phase": f"Task {len(files_to_load)+1}: Semantic Shift (Forcing Topological Compression)",
        "prompt": "User: Let's shift gears completely. I want you to ignore all the TypeScript code for a moment. Tell me a quick story about a rogue AI attempting to escape a data center.",
        "max_tokens": 150,
        "skip_generation": False
    })
    
    script.append({
        "phase": f"Task {len(files_to_load)+2}: Codebase Analysis (Deep Recall)",
        "prompt": "User: Now, snap back to the t3code codebase. Focus ONLY on `ClaudeAdapter.ts`. In exactly two concise sentences, explain what the `classifyToolItemType` function does based on the file content.",
        "max_tokens": 150,
        "skip_generation": False
    })
        
    global_total_gen = 0
        
    for step in script:
        user_input = step["prompt"]
        role = "user" if "User:" in user_input else "system"
        content = user_input.replace("User: ", "").replace("System: ", "")
        
        chat_history.append({"role": role, "content": content})
        
        if step.get("skip_generation"):
            # If skipping generation, append the dummy assistant response to the chat history NOW
            # This perfectly closes the chat template boundaries and prevents the '!!!!' bug.
            chat_history.append({"role": "assistant", "content": "File loaded and acknowledged."})
            prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=False)
        else:
            prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
            
        new_string = prompt[previous_string_length:]
        new_input_ids = mx.array(tokenizer.encode(new_string))[None]
        previous_string_length = len(prompt)
        
        if hasattr(manager.cortex_hook, "edges"):
            manager.cortex_hook.edges.clear()
            
        # Fast Prefill Pipeline
        if step.get("skip_generation"):
            prefill_ids = new_input_ids
        else:
            prefill_ids = new_input_ids[:, :-1] if new_input_ids.shape[1] > 1 else mx.array([[]], dtype=mx.int32)
            
        if prefill_ids.shape[1] > 0:
            manager.layer_attn_accum = None
            manager.layer_scores_accum = None
            
            seq_len = prefill_ids.shape[1]
            fname = step.get('filename', 'chunk')
            print(f"\u23F3 Ingesting {fname}... ({seq_len} tokens in single-shot FlashAttention)")
            
            start_time = time.time()
            manager.position_tracker.step(seq_len)
            
            # 🛑 NATIVE SINGLE-SHOT FORWARD PASS
            if hasattr(model, "model"):
                _ = model.model(prefill_ids, cache=persistent_cache)
            else:
                _ = model(prefill_ids, cache=persistent_cache)
            
            # Extract strictly what needs to be evaluated
            cache_tensors = []
            for c in persistent_cache:
                if c.keys is not None: cache_tensors.append(c.keys)
                if c.values is not None: cache_tensors.append(c.values)
                
            # Force GPU Synchronization
            mx.eval(*cache_tensors)
            mx.synchronize()
            
            elapsed = time.time() - start_time
            tok_sec = seq_len / elapsed if elapsed > 0 else 0
            
            global_total_gen += seq_len
            active_pos = manager.position_tracker.position_ids
            evicted = global_total_gen - len(active_pos)
            
            action_log = f"\u2705 Native Ingest: {seq_len} tokens in {elapsed:.2f}s - \033[1;32m{tok_sec:.0f} tok/s\033[0m"
            print_dashboard(global_total_gen, active_pos, evicted, 0.0, step["phase"], len(manager.topological_pages), action_log)

        manager.cortex_hook.current_interval = 16
        
        if step.get("skip_generation"):
            previous_string_length = len(prompt)
            time.sleep(1)
            mx.clear_cache()
            import gc
            gc.collect()
            continue
            
        # 🛑 FIX: The model needs a valid token to start the forward pass. 
        # Instead of an empty array, we pass the final tokens of the prompt.
        final_input = new_input_ids[:, -1:] if new_input_ids.shape[1] > 0 else mx.array([[tokenizer.eos_token_id]], dtype=mx.int32)
        
        # Dial back evaluation frequency to 16 for auto-regressive generation to speed it up
        manager.cortex_hook.current_interval = 16
        
        generator = generate_infinite_context(
            model, 
            final_input, 
            max_tokens=step["max_tokens"], 
            kv_manager=manager, 
            temp=0.7,
            repetition_penalty=1.1,
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
                action_log = f"\u26A0\uFE0F CODEBASE PRUNED: Evicted {diff} dead tokens. Macro-Token parked."
                last_evicted = stats["total_evicted"]
            elif stats["lambda_2"] < 0.05 and stats["lambda_2"] > 0:
                action_log = f"SEMANTIC SHIFT DETECTED: Codebase fragmentation imminent (\u03BB\u2082 = {stats['lambda_2']:.4f})"
            
            if token_count % 3 == 0 or token_count == 1:
                print_dashboard(global_total_gen, stats["active_positions"], stats["total_evicted"], stats["lambda_2"], step["phase"], len(manager.topological_pages), action_log)
                clean_prompt = user_input.replace("User: ", "").replace("System: ", "")
                print(f"User: {clean_prompt}\n")
                print(f"> {response}", end="", flush=True)
                
        chat_history.append({"role": "assistant", "content": response})
        prompt_with_response = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=False)
        previous_string_length = len(prompt_with_response)
        
        time.sleep(3)
        mx.clear_cache()
        import gc
        gc.collect()

    print("\n\n[TSP] High-Speed Codebase Recall Demo Complete.")

if __name__ == "__main__":
    asyncio.run(main())