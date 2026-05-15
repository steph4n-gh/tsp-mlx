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

def print_dashboard(total_gen, active_ids, evicted, lambda2, step_name, topological_pages=0, action_log=""):
    clear_screen()
    print(f"\033[1;37m[\u03C4-Spectral Pruner] MASSIVE CODEBASE DEMO\033[0m")
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
        print(f"\n\033[1;33m[VRAM PROTECTION SYSTEM] {action_log}\033[0m")
    
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
    print("Loading Coder Model (Qwen2.5-Coder-7B-Instruct-4bit)...")
    try:
        model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    except Exception as e:
        print(f"Failed to load model. Error: {e}")
        return

    # Setup aggressive pruning manager for the demo
    hook = CortexHook(eval_interval=5, threshold=0.99)
    manager = KVCacheManager(hook, model=model, enable_compression=True, enable_consolidation=True, head_dim=128)
    manager.consolidator.salience_threshold = 0.0 # Force TTT for demo
    model._tsp_kv_manager = manager

    chat_history = [
        {"role": "system", "content": "You are an expert AI Software Engineer. You read codebases and fix bugs."}
    ]
    
    # Simulate an agent "reading" a massive codebase file by file
    workflow_steps = [
        {
            "phase": "Reading Codebase (File 1: network.py)",
            "input": "System: Here is network.py for context.",
            "simulated_output": "Reading network.py...\nclass NetworkManager:\n    def connect(self):\n        pass\n    def disconnect(self):\n        pass\n    def send_packet(self, data):\n        # 500 lines of socket logic omitted for brevity\n        pass\n    def receive_packet(self):\n        pass\n\n> network.py loaded into context."
        },
        {
            "phase": "Reading Codebase (File 2: database.py)",
            "input": "System: Here is database.py for context.",
            "simulated_output": "Reading database.py...\nclass DatabaseConnector:\n    def __init__(self, uri):\n        self.uri = uri\n    def query(self, sql):\n        # 800 lines of SQL parsing logic omitted\n        pass\n    def flush(self):\n        pass\n\n> database.py loaded into context."
        },
        {
            "phase": "Reading Codebase (File 3: ui_components.tsx)",
            "input": "System: Here is ui_components.tsx for context.",
            "simulated_output": "Reading ui_components.tsx...\nexport const Button = () => <button>Click</button>;\nexport const Modal = () => <div>Modal</div>;\n// 1200 lines of React components omitted\n\n> ui_components.tsx loaded into context."
        },
        {
            "phase": "Fixing Bug in auth.py",
            "input": "User: There is a bug in auth.py where passwords are not hashed before saving. Ignore the rest of the codebase and write a secure implementation of the `register_user(username, password)` function.",
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
            _ = model(input_ids)
            
            seq_len = input_ids.shape[1]
            if not hasattr(manager.position_tracker, '_positions') or manager.position_tracker._positions.shape[0] == 0:
                 manager.position_tracker.step(seq_len)
            else:
                 manager.position_tracker.step(seq_len - manager.position_tracker._positions.shape[0])
            
            print_dashboard(seq_len, manager.position_tracker.position_ids, 0, 0.0, step["phase"], len(manager.topological_pages))
            print(f"{step['input']}\n")
            print(f"{step['simulated_output']}")
            time.sleep(2)
            continue

        # Real Generation Phase (Writing Code)
        prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
        input_ids = mx.array(tokenizer.encode(prompt))[None]
        
        generator = generate_infinite_context(model, input_ids, max_tokens=150)
        
        response = ""
        total_gen = input_ids.shape[1]
        last_evicted = 0
        action_log = ""
        
        print_dashboard(total_gen, manager.position_tracker.position_ids, 0, 0.0, step["phase"], len(manager.topological_pages))
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
                action_log = f"\u26A0\uFE0F CODEBASE PRUNED: Evicted {diff} tokens. Irrelevant files (Network, UI, DB) removed from VRAM."
                last_evicted = stats["total_evicted"]
            
            print_dashboard(total_gen, stats["active_positions"], stats["total_evicted"], stats["lambda_2"], step["phase"], len(manager.topological_pages), action_log)
            print(f"{step['input']}\n")
            print(f"> {response}", end="", flush=True)
            
        chat_history.append({"role": "assistant", "content": response})
        time.sleep(2)

    print("\n\n[TSP] Massive Codebase Demo Complete.")

if __name__ == "__main__":
    asyncio.run(main())
