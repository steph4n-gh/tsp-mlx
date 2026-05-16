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

def print_dashboard(total_gen, active_ids, evicted, lambda2, step_name, action_log=""):
    clear_screen()
    print(f"\033[1;37m[\u03C4-Spectral Pruner] AUTONOMOUS CLI AGENT WORKFLOW\033[0m")
    print(f"\033[1;36mCurrent Phase: {step_name}\033[0m")
    print("\033[1;30m-------------------------------------------------------------\033[0m")
    
    # Real-time Metrics
    print("\n\033[1;37mREAL-TIME VRAM METRICS:\033[0m")
    print(f"  \u25B6 \033[1mTotal Context History:\033[0m  {total_gen} tokens")
    print(f"  \u25B6 \033[1mActive Tokens In VRAM:\033[0m  {len(active_ids)} tokens")
    print(f"  \u25B6 \033[1mLog Noise Evicted:\033[0m      {evicted} tokens saved")
    
    if action_log:
        print(f"\n\033[1;33m[AGENT SYSTEM] {action_log}\033[0m")
    
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
    print("\n\033[1;37mTERMINAL STDOUT:\033[0m")

async def main():
    print("Loading model (Qwen2.5-Coder-7B-Instruct)...")
    try:
        model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-8bit")
    except Exception as e:
        print(f"Failed to load model. Error: {e}")
        return

    # Setup aggressive pruning manager for the demo
    hook = CortexHook(eval_interval=5, threshold=0.99)
    manager = KVCacheManager(hook, model=model, enable_compression=True, enable_consolidation=True, head_dim=128)
    manager.consolidator.salience_threshold = 0.0
    model._tsp_kv_manager = manager

    chat_history = [
        {"role": "system", "content": "You are Gemini CLI, an autonomous terminal agent. You run commands and write code."}
    ]
    
    # We simulate a workflow where the agent runs a noisy command, reads the output, 
    # and then moves on to write code.
    workflow_steps = [
        {
            "phase": "Executing Setup Command",
            "input": "User: Please set up the environment and then implement the new API route in server.ts.",
            "simulated_output": "> Executing: npm install\n\n[..................] - fetchMetadata: sill resolveWithNewModule yargs@17.7.2 checking installable status\n[..................] - fetchMetadata: sill resolveWithNewModule yargs@17.7.2 checking installable status\n[..................] - fetchMetadata: sill resolveWithNewModule yargs@17.7.2 checking installable status\n[..................] - fetchMetadata: sill resolveWithNewModule yargs@17.7.2 checking installable status\n[..................] - fetchMetadata: sill resolveWithNewModule yargs@17.7.2 checking installable status\n[..................] - fetchMetadata: sill resolveWithNewModule yargs@17.7.2 checking installable status\nadded 142 packages, and audited 143 packages in 3s\n\n> Setup complete."
        },
        {
            "phase": "Writing API Code",
            "input": "Agent Thought: The setup is done. I will now ignore the npm logs and focus entirely on writing a simple, short Express.js 'Hello World' route.",
            "simulated_output": "" # The model will generate this
        }
    ]
        
    for step in workflow_steps:
        chat_history.append({"role": "user", "content": step["input"]})
        
        # If we have simulated noisy output, we feed it as if the agent generated it
        if step["simulated_output"]:
            chat_history.append({"role": "assistant", "content": step["simulated_output"]})
            prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=False)
            input_ids = mx.array(tokenizer.encode(prompt))[None]
            
            # Fast forward the context window
            print(f"Simulating terminal output for {step['phase']}...")
            _ = model(input_ids)
            
            # Manually update the position tracker to match
            seq_len = input_ids.shape[1]
            if not hasattr(manager.position_tracker, '_positions') or manager.position_tracker._positions.shape[0] == 0:
                 manager.position_tracker.step(seq_len)
            else:
                 manager.position_tracker.step(seq_len - manager.position_tracker._positions.shape[0])
            
            print_dashboard(seq_len, manager.position_tracker.position_ids, 0, 0.0, step["phase"])
            print(f"{step['input']}\n")
            print(f"{step['simulated_output']}")
            time.sleep(3)
            continue

        # Real Generation Phase (Writing Code)
        prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
        input_ids = mx.array(tokenizer.encode(prompt))[None]
        
        generator = generate_infinite_context(model, input_ids, max_tokens=100)
        
        response = ""
        total_gen = input_ids.shape[1]
        last_evicted = 0
        action_log = ""
        
        print_dashboard(total_gen, manager.position_tracker.position_ids, 0, 0.0, step["phase"])
        print(f"{step['input']}\n")
        
        async for token, stats in generator:
            token_id = token.item()
            if token_id == tokenizer.eos_token_id:
                break
                
            text = tokenizer.decode([token_id])
            response += text
            total_gen += 1
            
            if stats["total_evicted"] > last_evicted:
                diff = stats["total_evicted"] - last_evicted
                action_log = f"\u26A0\uFE0F VRAM RECOVERY: Evicted {diff} tokens of dead terminal stdout from KV Cache."
                last_evicted = stats["total_evicted"]
            
            print_dashboard(total_gen, stats["active_positions"], stats["total_evicted"], stats["lambda_2"], step["phase"], action_log)
            print(f"{step['input']}\n")
            print(f"> {response}", end="", flush=True)
            
        chat_history.append({"role": "assistant", "content": response})
        time.sleep(2)

    print("\n\n[TSP] CLI Agent Demo Complete.")

if __name__ == "__main__":
    asyncio.run(main())
