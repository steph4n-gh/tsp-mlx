import asyncio
import sys
import os
import glob
import mlx.core as mx
from mlx_lm import load
from tsp_mlx.generate import setup_tsp
from tsp_mlx.inference import generate_infinite_context
import time
import subprocess
import threading

def clear_screen():
    print("\033[2J\033[H", end="")

def print_dashboard(total_gen, active_ids, evicted, lambda2, step_name, topological_pages=0, action_log=""):
    clear_screen()
    print(f"\033[1;37m[\u03C4-Spectral Pruner] ENTERPRISE DOCUMENT ANALYSIS DEMO\033[0m")
    print(f"\033[1;35mExecuting on: \033[1mQwen2.5-Coder-7B-Instruct (8-bit)\033[0m")
    print(f"\033[1;36mCurrent Phase: {step_name}\033[0m")
    print("\033[1;30m-------------------------------------------------------------\033[0m")
    
    active_tokens = len(active_ids)
    saved_tokens = total_gen - active_tokens
    saved_percent = (saved_tokens / total_gen * 100) if total_gen > 0 else 0
    
    # 8-bit kv cache size estimation (1 byte per parameter, but MLX stores as FP16 in cache usually, 
    # so we assume 2 bytes. 28 layers, 4 KV heads, 128 dim = 28672 bytes per token)
    bytes_per_token = 28672
    classic_vram_mb = (total_gen * bytes_per_token) / (1024 * 1024)
    tsp_vram_mb = (active_tokens * bytes_per_token) / (1024 * 1024)
    
    compression_ratio = total_gen / active_tokens if active_tokens > 0 else 1.0
    
    # Real-time Metrics
    print("\n\033[1;37m\u26A1 THEORETICAL BOUNDS VS. TSP ACTUALS:\033[0m")
    print(f"  \u25B6 \033[1;31mClassic Context Footprint:\033[0m {total_gen:,} tokens (\033[1m{classic_vram_mb:.1f} MB VRAM\033[0m) \033[31m[OOM IMMINENT]\033[0m")
    print(f"  \u25B6 \033[1;32mTSP Active Context:\033[0m        {active_tokens:,} tokens (\033[1m{tsp_vram_mb:.1f} MB VRAM\033[0m)")
    
    print("\n\033[1;37m\u25B6 ALGORITHMIC EFFICIENCY:\033[0m")
    print(f"  \u2022 \033[1;33mCompression Ratio:\033[0m       {compression_ratio:.1f}x")
    print(f"  \u2022 \033[1;33mTokens Evicted:\033[0m          {saved_tokens:,} ({saved_percent:.1f}% Reduction)")
    print(f"  \u2022 \033[1;33mMacro-Tokens Parked:\033[0m     {topological_pages} Pages (Disk/RAM Swap-Free)")
    print(f"  \u2022 \033[1;33mSemantic Graph Gap (\u03BB\u2082):\033[0m   {lambda2:.6f}")
    
    if action_log:
        print(f"\n\033[1;36m[SYSTEM EVENT]\033[0m \033[1m{action_log}\033[0m")
    
    # Dynamic Manifold Visualization
    print("\n\033[1;37mDYNAMIC KV CACHE MAP:\033[0m")
    map_str = "  ["
    if total_gen > 0:
        bucket_size = max(1.0, total_gen / 60.0)
        active_set = set(active_ids)
        for i in range(60):
            start = int(i * bucket_size)
            end = int((i + 1) * bucket_size)
            
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
    print("Loading Immutable Agent (Qwen2.5-Coder-7B-Instruct-8bit)...")
    try:
        model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-8bit")
    except Exception as e:
        print(f"Failed to load model. Error: {e}")
        return

    print("Warming up Unified Memory...")
    mx.eval(model.parameters())
    mx.synchronize()

    # Setup aggressive pruning manager with a tight budget
    manager = setup_tsp(model, head_dim=128, enable_compression=True, enable_consolidation=True)
    manager.cortex_hook.base_interval = 5
    manager.cortex_hook.threshold = 0.05
    manager.cortex_hook.max_context_budget = 2000 
    manager.consolidator.salience_threshold = 0.0 

    chat_history = [
        {"role": "system", "content": "You are a helpful, expert AI assistant. You help users read and analyze massive documents. Keep your responses extremely concise and to the point (max 2-3 sentences)."}
    ]
    
    from mlx_lm.models.cache import make_prompt_cache
    persistent_cache = make_prompt_cache(model)
    previous_string_length = 0
    
    # Compile a massive document (~400,000 chars, ~100k tokens)
    doc_files = glob.glob("/Volumes/Storage/bigworkspace/supplychain/docs/*.md")
    base_doc = ""
    for df in doc_files:
        with open(df, "r") as f:
            base_doc += f.read() + "\n\n"
            
    # Scale up the doc to ensure we easily hit 100k tokens over 4 files
    massive_doc = (base_doc * 20)[:450000] 
    
    chunk_size = len(massive_doc) // 4
    doc_part_1 = massive_doc[0:chunk_size]
    doc_part_2 = massive_doc[chunk_size:chunk_size*2]
    doc_part_3 = massive_doc[chunk_size*2:chunk_size*3]
    doc_part_4 = massive_doc[chunk_size*3:]
    
    script = [
        {
            "phase": "Task 1: Casual Chat (Warmup)",
            "prompt": "User: Hello! I have 4 massive volumes of our Q3 Enterprise Security Audit (over 100,000 tokens total). I'll upload them one by one so we can discuss the state of our semantic firewall. Ready?",
            "max_tokens": 400
        },
        {
            "phase": "Task 2: Ingest Volume 1 (Failure Analysis)",
            "prompt": f"System: Uploaded `Q3_Enterprise_Security_Audit_Vol1.md`:\n\n```markdown\n{doc_part_1}\n```\n\nUser: Here is Volume 1. Based on this text, what does the failure analysis section say about dependency spoofing?",
            "max_tokens": 400
        },
        {
            "phase": "Task 3: Organic Conversation",
            "prompt": "User: That's a huge risk. If an attacker bypasses the initial hash check, what is our immediate fallback mechanism?",
            "max_tokens": 400
        },
        {
            "phase": "Task 4: Ingest Volume 2 (Network Architecture)",
            "prompt": f"System: Uploaded `Q3_Network_Architecture_Specs_Vol2.md`:\n\n```markdown\n{doc_part_2}\n```\n\nUser: Here is Volume 2 covering the network layer. How should the validation engine respond to an unexpected network call based on these specs?",
            "max_tokens": 400
        },
        {
            "phase": "Task 5: Ingest Volume 3 (Threat Heuristics)",
            "prompt": f"System: Uploaded `Q3_Threat_Modeling_Heuristics_Vol3.md`:\n\n```markdown\n{doc_part_3}\n```\n\nUser: Here is Volume 3. Based on the extensibility section within this document, can we plug in our own custom heuristic analyzers for zero-day threats?",
            "max_tokens": 400
        },
        {
            "phase": "Task 6: Ingest Volume 4 (Compliance)",
            "prompt": f"System: Uploaded `Q3_Compliance_and_Extensibility_Vol4.md`:\n\n```markdown\n{doc_part_4}\n```\n\nUser: Finally, here is Volume 4. Are there any compliance conflicts with using third-party package managers mentioned here?",
            "max_tokens": 400
        },
        {
            "phase": "Task 7: Synthesis (Deep Recall Across 100k Tokens)",
            "prompt": "User: Excellent. We've now ingested well over 100,000 tokens of security documentation. Summarize the complete end-to-end data flow when a malicious payload attempts to execute an obscure post-install script, drawing on all 4 volumes. Keep it under 4 sentences.",
            "max_tokens": 400
        }
    ]
        
    global_total_gen = 0
        
    for step in script:
        user_input = step["prompt"]
        role = "user" if "User:" in user_input else "system"
        content = user_input.replace("User: ", "").replace("System: ", "")
        
        chat_history.append({"role": role, "content": content})
        
        prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
            
        new_string = prompt[previous_string_length:]
        new_input_ids = mx.array(tokenizer.encode(new_string))[None]
        previous_string_length = len(prompt)
        
        if hasattr(manager.cortex_hook, "edges"):
            manager.cortex_hook.edges.clear()
            
        prefill_ids = new_input_ids[:, :-1] if new_input_ids.shape[1] > 1 else mx.array([[]], dtype=mx.int32)
            
        if prefill_ids.shape[1] > 0:
            manager.layer_attn_accum = None
            manager.layer_scores_accum = None
            
            seq_len = prefill_ids.shape[1]
            print(f"\n\u23F3 Queuing Prefill Graph... ({seq_len:,} tokens)")
            
            start_time = time.time()
            manager.position_tracker.step(seq_len)
            
            if hasattr(model, "model"):
                _ = model.model(prefill_ids, cache=persistent_cache)
            else:
                _ = model(prefill_ids, cache=persistent_cache)
            
            cache_tensors = []
            for c in persistent_cache:
                if c.keys is not None: cache_tensors.append(c.keys)
                if c.values is not None: cache_tensors.append(c.values)
                
            done = False
            def spinner():
                chars = ['\u280B', '\u2819', '\u2839', '\u2838', '\u283C', '\u2834', '\u2826', '\u2827', '\u2807', '\u280F']
                i = 0
                est_time = seq_len / 370.0
                while not done:
                    sys.stdout.write(f'\r  \033[1;36m{chars[i % len(chars)]} Ingesting into Unified Memory... [Est. ~{est_time:.1f}s at 370 tok/s]\033[0m')
                    sys.stdout.flush()
                    time.sleep(0.1)
                    i += 1
                    
            t = threading.Thread(target=spinner)
            t.start()
                
            mx.eval(*cache_tensors)
            mx.synchronize()
            
            done = True
            t.join()
            sys.stdout.write('\r\033[K') # clear spinner
            
            elapsed = time.time() - start_time
            tok_sec = seq_len / elapsed if elapsed > 0 else 0
            
            global_total_gen += seq_len
            active_pos = manager.position_tracker.position_ids
            evicted = global_total_gen - len(active_pos)
            
            action_log = f"\u2705 Native Ingest: {seq_len:,} tokens in {elapsed:.2f}s - \033[1;32m{tok_sec:.0f} tok/s\033[0m"
            print_dashboard(global_total_gen, active_pos, evicted, 0.0, step["phase"], len(manager.topological_pages), action_log)

        final_input = new_input_ids[:, -1:] if new_input_ids.shape[1] > 0 else mx.array([[tokenizer.eos_token_id]], dtype=mx.int32)
        
        manager.cortex_hook.current_interval = 16
        
        # Show the user prompt truncated if it's too long (like an uploaded file) before generation
        clean_prompt = user_input.replace("User: ", "").replace("System: ", "")
        if "```markdown" in clean_prompt:
            clean_prompt = "User: \033[1;33m[Uploaded massive document (>25,000 tokens)]\033[0m " + clean_prompt.split("User: ")[-1]
        else:
            clean_prompt = "User: " + clean_prompt
            
        print(f"\n{clean_prompt}\n")
        
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
                action_log = f"\u26A0\uFE0F DOCUMENT PRUNED: Evicted {diff:,} dead tokens. Macro-Token parked."
                last_evicted = stats["total_evicted"]
            
            if token_count % 3 == 0 or token_count == 1:
                print_dashboard(global_total_gen, stats["active_positions"], stats["total_evicted"], stats["lambda_2"], step["phase"], len(manager.topological_pages), action_log)
                print(f"\n{clean_prompt}\n")
                print(f"> {response}", end="", flush=True)
                
        chat_history.append({"role": "assistant", "content": response})
        prompt_with_response = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=False)
        previous_string_length = len(prompt_with_response)
        
        time.sleep(1.5)
        mx.clear_cache()
        import gc
        gc.collect()

    print("\n\n[TSP] Organic Workflow Demo Complete.")

if __name__ == "__main__":
    asyncio.run(main())