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

def print_ui(total_gen, active_ids, evicted, action_log=""):
    clear_screen()
    print("\033[1;36m=================================================================\033[0m")
    print("\033[1;36m  \u26A1 GUI Orchestrator Agent | Powered by \u03C4-Spectral Pruner  \033[0m")
    print("\033[1;36m=================================================================\033[0m\n")
    
    # TSP Backend HUD (Invisible to normal user, exposed for demo)
    print("\033[1;30m[BACKEND VRAM METRICS]\033[0m")
    print(f"\033[1;30m  > Total Context Generated:  {total_gen} tokens\033[0m")
    print(f"\033[1;30m  > Active VRAM Footprint:    {len(active_ids)} tokens\033[0m")
    
    if action_log:
         print(f"\033[1;35m  > {action_log}\033[0m")
    else:
         print(f"\033[1;30m  > Total Noise Pruned:       {evicted} tokens saved\033[0m")
         
    # Visual Map
    map_str = "\033[1;30m  > MAP: ["
    for i in range(min(50, len(active_ids))):
        if i < 3: map_str += "\033[32m\u2588\033[30m" # Sinks
        elif i > len(active_ids) - 15: map_str += "\033[36m\u2592\033[30m" # Active
        else: map_str += "\u2591" # Persistent
    map_str += "]\033[0m"
    print(map_str)
    
    print("\n\033[1;37m[TERMINAL]\033[0m")

async def main():
    print("Initializing GUI Orchestrator Agent with Qwen2.5-Coder-7B-Instruct...")
    try:
        model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    except Exception as e:
        print(f"Failed to load model. Error: {e}")
        return

    # Initialize the TSP backend
    hook = CortexHook(eval_interval=5, threshold=0.99)
    manager = KVCacheManager(hook, model=model, enable_compression=True, enable_consolidation=True, head_dim=128)
    manager.consolidator.salience_threshold = 0.0 # Force TTT for demo
    model._tsp_kv_manager = manager

    chat_history = [
        {"role": "system", "content": "You are an autonomous CLI agent orchestrated by a GUI. You read commands, write code, and scaffold applications."}
    ]

    # GUI Orchestrator Workflow Simulation
    workflow_steps = [
        {
            "phase": "User Input",
            "input": "User: Scaffold a new Next.js app with Tailwind and create a login page.",
            "type": "user"
        },
        {
            "phase": "Executing Command",
            "input": "",
            "simulated_output": "$ npx create-next-app@latest my-app --typescript --tailwind --eslint\nCreating a new Next.js app in /my-app.\n\nUsing npm.\nInitializing project with template: app-tw\nInstalling dependencies:\n- react\n- react-dom\n- next\n- tailwindcss\n- postcss\n- autoprefixer\n- eslint\n- eslint-config-next\n\nadded 362 packages, and audited 363 packages in 12s\n106 packages are looking for funding\n  run `npm fund` for details\n\nSuccess! Created my-app at /my-app",
            "type": "command_output"
        },
        {
            "phase": "Generating Code",
            "input": "Agent: The scaffolding is complete. I will now ignore the installation logs and write the code for the Login page in app/login/page.tsx.",
            "type": "agent_action"
        }
    ]
        
    for step in workflow_steps:
        if step["type"] == "user":
             chat_history.append({"role": "user", "content": step["input"]})
             continue
             
        elif step["type"] == "command_output":
            chat_history.append({"role": "assistant", "content": step["simulated_output"]})
            prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=False)
            input_ids = mx.array(tokenizer.encode(prompt))[None]
            
            # Fast forward context (Simulate the agent reading the terminal output)
            _ = model(input_ids)
            
            seq_len = input_ids.shape[1]
            if not hasattr(manager.position_tracker, '_positions') or manager.position_tracker._positions.shape[0] == 0:
                 manager.position_tracker.step(seq_len)
            else:
                 manager.position_tracker.step(seq_len - manager.position_tracker._positions.shape[0])
            
            print_ui(seq_len, manager.position_tracker.position_ids, 0)
            print(f"\033[1;32m{step['simulated_output']}\033[0m")
            time.sleep(3)
            continue

        elif step["type"] == "agent_action":
            chat_history.append({"role": "user", "content": step["input"]})
            prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
            input_ids = mx.array(tokenizer.encode(prompt))[None]
            
            generator = generate_infinite_context(model, input_ids, max_tokens=150)
            
            response = ""
            total_gen = input_ids.shape[1]
            last_evicted = 0
            action_log = ""
            
            print_ui(total_gen, manager.position_tracker.position_ids, 0)
            print(f"\033[1;36m{step['input']}\033[0m\n")
            
            async for token, stats in generator:
                token_id = token.item()
                if token_id == tokenizer.eos_token_id:
                    break
                    
                text = tokenizer.decode([token_id])
                response += text
                total_gen += 1
                
                if stats["total_evicted"] > last_evicted:
                    diff = stats["total_evicted"] - last_evicted
                    action_log = f"\u26A0\uFE0F TSP BACKEND: Pruned {diff} tokens of raw NPM installation logs from VRAM."
                    last_evicted = stats["total_evicted"]
                
                print_ui(total_gen, stats["active_positions"], stats["total_evicted"], action_log)
                print(f"\033[1;36m{step['input']}\033[0m\n")
                print(f"{response}", end="", flush=True)
                
            chat_history.append({"role": "assistant", "content": response})
            time.sleep(2)

    print("\n\n\033[1;32m[DONE] GUI Orchestrator integration demo completed successfully.\033[0m")

if __name__ == "__main__":
    asyncio.run(main())
