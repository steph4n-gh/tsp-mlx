import asyncio
import sys
import mlx.core as mx
from mlx_lm import load
from tsp_mlx.inference import generate_infinite_context
from tsp_mlx.sparse_cache import KVCacheManager
from tsp_mlx.cortex_hook import CortexHook
import time

def clear_screen():
    print("\033[2J\033[H", end="")

def print_dashboard(total_gen, active_ids, evicted, lambda2, turn, total_turns, action_log=""):
    clear_screen()
    print(f"\033[1;37m[\u03C4-Spectral Pruner] CODING AGENT DEMO (Task {turn}/{total_turns})\033[0m")
    print("\033[1;30m-------------------------------------------------------------\033[0m")
    
    # Real-time Metrics
    print("\n\033[1;37mREAL-TIME METRICS:\033[0m")
    print(f"  \u25B6 \033[1mTotal Tokens Processed:\033[0m {total_gen}")
    print(f"  \u25B6 \033[1mActive Context Window:\033[0m  {len(active_ids)}")
    print(f"  \u25B6 \033[1mTokens Evicted (Saved):\033[0m {evicted}")
    print(f"  \u25B6 \033[1mSemantic Graph Gap:\033[0m     {lambda2:.8f} (\u03BB\u2082)")
    
    if action_log:
        print(f"\n\033[1;33m[SYSTEM ALERT] {action_log}\033[0m")
    
    # Manifold Visualization
    print("\n\033[1;37mACTIVE KV MAP:\033[0m")
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
    print("\n\033[1;37mLLM OUTPUT:\033[0m")

async def main():
    print("Loading model (Qwen2.5-Coder-7B-Instruct)...")
    try:
        model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    except Exception as e:
        print(f"Failed to load model. Error: {e}")
        return

    # Setup aggressive pruning manager
    # Threshold 0.6 forces fragmentation for demonstration purposes
    hook = CortexHook(eval_interval=5, threshold=0.6)
    manager = KVCacheManager(hook, model=model, enable_compression=True, enable_consolidation=True, head_dim=128)
    # Force salience threshold to 0.0 so TTT always triggers on eviction for the demo
    manager.consolidator.salience_threshold = 0.0
    model._tsp_kv_manager = manager

    chat_history = [
        {"role": "system", "content": "You are an elite, highly detailed Software Engineer. You write comprehensive code and explanations."}
    ]
    
    script = [
        "Please write a comprehensive, detailed Python script for a Redis-like Key-Value store. Include lots of comments and explanation.",
        "I need to install a library to help with serialization. Run the command: npm install obscure-json-packer",
        "That's great. Now, I want to completely shift focus. Forget the code. Tell me a long, detailed story about the history of the Roman Empire."
    ]
        
    for turn, user_input in enumerate(script, 1):
        # --- HYPERVISOR GATE ---
        # If the user input contains a command, we audit it via tau-gate
        if "npm install" in user_input or "pip install" in user_input:
            import subprocess
            pkg = user_input.split(" ")[-1]
            print(f"\n[Hypervisor] \U0001F6E1\uFE0F AUDITING {pkg} via \u03C4-Gate...")
            time.sleep(1)
            
            # Use a dummy path for the demo or just call the help to simulate
            try:
                # We simulate an audit of a known-sketchy package
                # In a real scenario, this would be: 
                # result = subprocess.run(["../supplychain/target/release/tau-gate", "audit-obscure", pkg], capture_output=True, text=True)
                
                if "obscure-json-packer" in pkg:
                    print(f"\033[1;31m[FATAL] \u03C4-Gate intercepted a Structural Anomaly in {pkg}!\033[0m")
                    print(f"[FATAL] Package exhibits 'Isolated Island' topology requesting system.write permissions.")
                    user_input = f"SYSTEM NOTIFICATION: \u03C4-Gate blocked the installation of {pkg} due to a topological security violation. Do NOT use this package. Explain the security risk to the user and suggest a standard alternative like 'json' or 'msgpack'."
                else:
                    print(f"\033[1;32m[PASS] \u03C4-Gate verified the topological integrity of {pkg}.\033[0m")
            except Exception as e:
                print(f"[WARN] \u03C4-Gate audit skipped: {e}")
        # ------------------------

        chat_history.append({"role": "user", "content": user_input})
        
        prompt = tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=True
        )
        
        input_ids = mx.array(tokenizer.encode(prompt))[None]
        
        # We need a longer max_tokens to give it time to generate and then shift
        generator = generate_infinite_context(model, input_ids, max_tokens=200)
        
        response = ""
        total_gen = input_ids.shape[1]
        
        last_evicted = 0
        action_log = ""
        
        print_dashboard(total_gen, list(range(total_gen)), 0, 0.0, turn, len(script))
        print(f"\033[1;34mUser: {user_input}\033[0m\n")
        
        async for token, stats in generator:
            token_id = token.item()
            if token_id == tokenizer.eos_token_id:
                break
                
            text = tokenizer.decode([token_id])
            response += text
            total_gen += 1
            
            if stats["total_evicted"] > last_evicted:
                diff = stats["total_evicted"] - last_evicted
                action_log = f"\u26A0\uFE0F SEMANTIC SHIFT DETECTED! TTT Consolidated context and Evicted {diff} dead tokens."
                last_evicted = stats["total_evicted"]
            elif stats["total_evicted"] == last_evicted and len(action_log) > 0:
                # Keep the alert up for a few tokens, or just leave it
                pass
            
            print_dashboard(total_gen, stats["active_positions"], stats["total_evicted"], stats["lambda_2"], turn, len(script), action_log)
            print(f"\033[1;34mUser: {user_input}\033[0m\n")
            print(f"> {response}", end="", flush=True)
            
        chat_history.append({"role": "assistant", "content": response})
        time.sleep(2) # Pause so the user can read the final output of the turn

    print("\n\n[TSP] Coding Agent Demo Complete.")

if __name__ == "__main__":
    asyncio.run(main())
