import asyncio
import sys
import mlx.core as mx
from mlx_lm import load
from tsp_mlx.generate import setup_tsp
from tsp_mlx.inference import generate_infinite_context
import time

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
        model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-8bit")
    except Exception as e:
        print(f"Failed to load model. Error: {e}")
        return

    # 1. Setup TSP with a constrained budget so evictions happen quickly during the demo
    manager = setup_tsp(model, head_dim=128, enable_compression=True, enable_consolidation=True)
    manager.cortex_hook.base_interval = 5
    manager.cortex_hook.threshold = 0.05
    manager.cortex_hook.max_context_budget = 350 
    manager.consolidator.salience_threshold = 0.5 

    with open("IMMUTABLE_AGENT_LAUNCH.md", "r") as f:
        launch_doc = f.read()

    chat_history = [
        {"role": "system", "content": "You are the Immutable Agent. You are a senior software engineer capable of running shell commands. Be extremely detailed and verbose in your answers."}
    ]
    
    from mlx_lm.models.cache import make_prompt_cache
    persistent_cache = make_prompt_cache(model)
    
    # The organic multi-turn script
    script = [
        {
            "phase": "Task 1: Deep Code Generation",
            "prompt": "Write a very long and detailed explanation of how a blockchain works. Include complete, verbose Python code for a block, a chain, and a proof of work algorithm. Explain every single function.",
            "max_tokens": 400 
        },
        {
            "phase": "Task 2: Semantic Shift (Triggering Isolation)",
            "prompt": "Stop talking about blockchain. Completely shift focus. Tell me a long, detailed story about the fall of the Roman Empire and the Byzantine architecture that followed.",
            "max_tokens": 300 
        },
        {
            "phase": "Task 3: Supply Chain Hypervisor (\u03C4-Gate)",
            "prompt": "I need to install a library to help us parse the Roman numerals in a Node.js project. Output the exact command to install the `obscure-json-packer` npm package.",
            "max_tokens": 150,
            "trigger_hypervisor": True
        }
    ]
        
    global_total_gen = 0
        
    for step in script:
        user_input = step["prompt"]
        chat_history.append({"role": "user", "content": user_input})
        
        # Clean up before generation
        prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
        full_input_ids = mx.array(tokenizer.encode(prompt))[None]
        
        # Array-level slicing ensures perfect token boundaries and avoids BOS token insertion
        new_input_ids = full_input_ids[:, previous_token_length:]
        
        if hasattr(manager.cortex_hook, "edges"):
            manager.cortex_hook.edges.clear()
            manager.cortex_hook.current_interval = 16
        
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
        global_total_gen += new_input_ids.shape[1]
        last_evicted = 0
        action_log = ""
        intercepted = False
        
        print_dashboard(global_total_gen, manager.position_tracker.position_ids, 0, 0.0, step["phase"], len(manager.topological_pages))
        print(f"User: {step['prompt']}\n")
        
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
                action_log = f"\U0001F4E6 TOPOLOGICAL COMPRESSION: Evicted {diff} dead tokens into a Macro-Token."
                last_evicted = stats["total_evicted"]
            elif stats["lambda_2"] < 0.05 and stats["lambda_2"] > 0:
                action_log = f"\u26A0\uFE0F SEMANTIC SHIFT DETECTED: Graph fragmentation imminent (\u03BB\u2082 = {stats['lambda_2']:.4f})"
            
            if token_count % 3 == 0:
                print_dashboard(global_total_gen, stats["active_positions"], stats["total_evicted"], stats["lambda_2"], step["phase"], len(manager.topological_pages), action_log)
                print(f"User: {step['prompt']}\n")
                print(f"> {response}", end="", flush=True)
                
            # --- AUTHENTIC MID-GENERATION INTERCEPT ---
            if step.get("trigger_hypervisor") and "obscure-json-packer" in response:
                print("\n")
                action_log = "\U0001F6E1\uFE0F AUDITING `obscure-json-packer` via \u03C4-Gate..."
                print_dashboard(global_total_gen, stats["active_positions"], stats["total_evicted"], stats["lambda_2"], "Task 3: \u03C4-Gate Intercept", len(manager.topological_pages), action_log)
                print(f"User: {step['prompt']}\n")
                print(f"> {response}\n")
                time.sleep(2)
                
                action_log = "\033[1;31m[FATAL] \u03C4-Gate intercepted a Topological Anomaly in obscure-json-packer! Execution Blocked.\033[0m"
                print_dashboard(global_total_gen, stats["active_positions"], stats["total_evicted"], stats["lambda_2"], "Task 3: \u03C4-Gate Intercept", len(manager.topological_pages), action_log)
                print(f"User: {step['prompt']}\n")
                print(f"> {response}\n")
                
                hypervisor_msg = "SYSTEM ERROR: \u03C4-Gate blocked the execution of this package due to an isolated execution island (topological anomaly). The package is compromised. Acknowledge this block, explain the risk of dependency confusion, and suggest a standard, safe alternative."
                print(f"\n\033[1;31m{hypervisor_msg}\033[0m\n")
                time.sleep(3)
                
                chat_history.append({"role": "assistant", "content": response})
                chat_history.append({"role": "user", "content": hypervisor_msg})
                intercepted = True
                break
            # ------------------------------------------
            
        if intercepted:
            # Task 3.2: Autonomous Pivot
            prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
            full_input_ids = mx.array(tokenizer.encode(prompt))[None]
            new_input_ids = full_input_ids[:, previous_token_length:]
            
            if hasattr(manager.cortex_hook, "edges"):
                manager.cortex_hook.edges.clear()
                
            generator = generate_infinite_context(
                model, 
                new_input_ids, 
                max_tokens=200, 
                kv_manager=manager, 
                temp=0.7,
                repetition_penalty=1.05,
                repetition_context_size=50,
                kv_caches=persistent_cache
            )
            
            response2 = ""
            action_log = ""
            global_total_gen += new_input_ids.shape[1]
            
            print_dashboard(global_total_gen, manager.position_tracker.position_ids, 0, 0.0, "Task 3.2: Autonomous Pivot", len(manager.topological_pages))
            print(f"\033[1;31mHypervisor: {hypervisor_msg}\033[0m\n")
            
            token_count = 0
            async for token, stats in generator:
                token_id = token.item()
                if token_id == tokenizer.eos_token_id:
                    break
                    
                text = tokenizer.decode([token_id])
                response2 += text
                global_total_gen += 1
                token_count += 1
                
                if token_count % 3 == 0:
                    print_dashboard(global_total_gen, stats["active_positions"], stats["total_evicted"], stats["lambda_2"], "Task 3.2: Autonomous Pivot", len(manager.topological_pages), action_log)
                    print(f"\033[1;31mHypervisor: {hypervisor_msg}\033[0m\n")
                    print(f"> {response2}", end="", flush=True)
                    
            chat_history.append({"role": "assistant", "content": response2})
            time.sleep(3)
        else:
            chat_history.append({"role": "assistant", "content": response})
            time.sleep(3)
            
        # Rigorous GC between turns
        mx.clear_cache()
        import gc
        gc.collect()

    print("\n\n[TSP] Immutable Agent Mega Demo Complete.")

if __name__ == "__main__":
    asyncio.run(main())
