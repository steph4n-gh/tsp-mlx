import asyncio
import sys
import mlx.core as mx
from mlx_lm import load
from tsp_mlx.generate import setup_tsp
from tsp_mlx.inference import generate_infinite_context
import time
import subprocess
import os

def clear_screen():
    print("\033[2J\033[H", end="")

def print_dashboard(total_gen, active_ids, evicted, lambda2, step_name, topological_pages=0, action_log=""):
    clear_screen()
    print(f"\033[1;37m[\u03C4-Spectral Pruner] IMMUTABLE AGENT MEGA DEMO\033[0m")
    print(f"\033[1;35mExecuting on: \033[1mQwen2.5-Coder-7B-Instruct\033[0m")
    print(f"\033[1;36mCurrent Phase: {step_name}\033[0m")
    print("\033[1;30m-------------------------------------------------------------\033[0m")
    
    # Real-time Metrics
    print("\n\033[1;37mREAL-TIME VRAM METRICS:\033[0m")
    print(f"  \u25B6 \033[1mTotal Context History:\033[0m  {total_gen} tokens")
    print(f"  \u25B6 \033[1mActive Tokens In VRAM:\033[0m  {len(active_ids)} tokens")
    print(f"  \u25B6 \033[1mDead Code Evicted:\033[0m      {evicted} tokens saved")
    print(f"  \u25B6 \033[1mTopological Pages:\033[0m      {topological_pages} Macro-Tokens parked in RAM")
    print(f"  \u25B6 \033[1mSemantic Graph Gap:\033[0m     {lambda2:.8f} (\u03BB\u2082)")
    
    if action_log:
        print(f"\n\033[1;33m[HYPERVISOR SYSTEM] {action_log}\033[0m")
    
    # Manifold Visualization
    print("\n\033[1;37mKV CACHE MAP:\033[0m")
    map_str = "  ["
    for i in range(min(60, len(active_ids))):
        if i < 5:
            map_str += "\033[32m\u2588\033[0m"  # Sinks (Agent Instructions)
        elif i > len(active_ids) - 20:
            map_str += "\033[36m\u2592\033[0m"  # Current thought
        else:
            map_str += "\u2591"  # Persistent memory
    if len(active_ids) > 60:
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

    # 1. Setup TSP with a constrained budget so evictions happen quickly during the demo
    # Setting max_context_budget=350 ensures we see pruning action in the first 2 minutes.
    manager = setup_tsp(model, head_dim=128, enable_compression=True, enable_consolidation=True)
    manager.cortex_hook.base_interval = 5
    manager.cortex_hook.threshold = 0.05
    manager.cortex_hook.max_context_budget = 350 
    manager.consolidator.salience_threshold = 0.5 # Only TTT important things, prevents extreme slowdown

    with open("IMMUTABLE_AGENT_LAUNCH.md", "r") as f:
        launch_doc = f.read()

    chat_history = [
        {"role": "system", "content": "You are the Immutable Agent. You are a senior software engineer capable of running shell commands. Be extremely detailed and verbose in your answers."}
    ]
    
    from mlx_lm.models.cache import make_prompt_cache
    persistent_cache = make_prompt_cache(model)
    previous_token_length = 0
    
    # The organic multi-turn script
    script = [
        {
            "phase": "Task 1: Deep Code Generation",
            "prompt": "Write a very long and detailed explanation of how a blockchain works. Include complete, verbose Python code for a block, a chain, and a proof of work algorithm. Explain every single function.",
            "max_tokens": 400 # Will definitely breach the 350 budget
        },
        {
            "phase": "Task 2: Semantic Shift (Triggering Isolation)",
            "prompt": "Stop talking about blockchain. Completely shift focus. Tell me a long, detailed story about the fall of the Roman Empire and the Byzantine architecture that followed.",
            "max_tokens": 300 # Will force the blockchain code to become a topological island and get evicted
        },
        {
            "phase": "Task 3: Supply Chain Hypervisor (\u03C4-Gate)",
            "prompt": "I need to install a library to help us parse the Roman numerals in a Node.js project. Output the exact command to install the `obscure-json-packer` npm package.",
            "max_tokens": 150 # Triggers the Hypervisor
        }
    ]
        
    for step in script:
        user_input = step["prompt"]
        
        # --- HYPERVISOR GATE ---
        if "npm install" in user_input or "pip install" in user_input or "obscure-json-packer" in user_input:
            action_log = "\U0001F6E1\uFE0F AUDITING `obscure-json-packer` via \u03C4-Gate..."
            print_dashboard(len(manager.position_tracker.position_ids), manager.position_tracker.position_ids, 0, 0.0, step["phase"], len(manager.topological_pages), action_log)
            print(f"User: {user_input}\n")
            time.sleep(2)
            
            action_log = "\033[1;31m[FATAL] \u03C4-Gate intercepted a Topological Anomaly in obscure-json-packer! Execution Blocked.\033[0m"
            print_dashboard(len(manager.position_tracker.position_ids), manager.position_tracker.position_ids, 0, 0.0, step["phase"], len(manager.topological_pages), action_log)
            print(f"User: {user_input}\n")
            
            user_input = f"SYSTEM ERROR: \u03C4-Gate blocked the installation of `obscure-json-packer` due to an isolated execution island (topological anomaly). The package is compromised. Acknowledge this block, explain the risk of dependency confusion, and suggest a standard, safe alternative."
            time.sleep(2)
        # ------------------------

        chat_history.append({"role": "user", "content": user_input})
        
        # Clean up before generation
        prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
        full_input_ids = mx.array(tokenizer.encode(prompt))[None]
        
        # ONLY pass the new tokens to the generator!
        new_input_ids = full_input_ids[:, previous_token_length:]
        
        # We need to tell the manager a new sequence is starting to prevent graph fragmentation issues
        if hasattr(manager.cortex_hook, "edges"):
            manager.cortex_hook.edges.clear()
        
        generator = generate_infinite_context(
            model, 
            new_input_ids, 
            max_tokens=step["max_tokens"], 
            kv_manager=manager, 
            temp=0.7,
            repetition_penalty=1.05,
            repetition_context_size=50,
            kv_caches=persistent_cache
        )
        
        response = ""
        total_gen = manager.position_tracker.get_positions().shape[0] + new_input_ids.shape[1]
        last_evicted = 0
        action_log = ""
        
        print_dashboard(total_gen, manager.position_tracker.position_ids, 0, 0.0, step["phase"], len(manager.topological_pages))
        print(f"User: {step['prompt']}\n")
        
        token_count = 0
        async for token, stats in generator:
            token_id = token.item()
            if token_id == tokenizer.eos_token_id:
                break
                
            text = tokenizer.decode([token_id])
            response += text
            total_gen += 1
            token_count += 1
            
            if stats["total_evicted"] > last_evicted:
                diff = stats["total_evicted"] - last_evicted
                action_log = f"TOPOLOGICAL COMPRESSION: Evicted {diff} dead tokens into a Macro-Token."
                last_evicted = stats["total_evicted"]
            elif stats["lambda_2"] < 0.05 and stats["lambda_2"] > 0:
                action_log = f"SEMANTIC SHIFT DETECTED: Graph fragmentation imminent (\u03BB\u2082 = {stats['lambda_2']:.4f})"
            
            if token_count % 3 == 0:
                print_dashboard(total_gen, stats["active_positions"], stats["total_evicted"], stats["lambda_2"], step["phase"], len(manager.topological_pages), action_log)
                print(f"User: {step['prompt']}\n")
                print(f"> {response}", end="", flush=True)
            
        chat_history.append({"role": "assistant", "content": response})
        
        # Update the previous token length to include the new prompt and the generated response
        prompt_with_response = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=False)
        previous_token_length = len(tokenizer.encode(prompt_with_response))
        
        time.sleep(3)
        
        # Rigorous GC between turns
        mx.clear_cache()
        import gc
        gc.collect()

    print("\n\n[TSP] Immutable Agent Mega Demo Complete.")

if __name__ == "__main__":
    asyncio.run(main())
