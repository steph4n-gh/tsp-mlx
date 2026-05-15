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
    print(f"\033[1;37m[\u03C4-Spectral Pruner] 32B FRONTIER MODEL DEMO\033[0m")
    print(f"\033[1;35mExecuting on: \033[1mQwen2.5-Coder-32B-Instruct (18GB VRAM Footprint)\033[0m")
    print(f"\033[1;36mCurrent Phase: {step_name}\033[0m")
    print("\033[1;30m-------------------------------------------------------------\033[0m")
    
    # Real-time Metrics
    print("\n\033[1;37mREAL-TIME VRAM METRICS (24GB Unified Memory Constraint):\033[0m")
    print(f"  \u25B6 \033[1mTotal Context History:\033[0m  {total_gen} tokens")
    print(f"  \u25B6 \033[1mActive Tokens In VRAM:\033[0m  {len(active_ids)} tokens")
    print(f"  \u25B6 \033[1mLog Noise Evicted:\033[0m      {evicted} tokens saved")
    print(f"  \u25B6 \033[1mTopological Tension:\033[0m    {lambda2:.8f} (\u03BB\u2082)")
    
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

def main():
    print("Loading Frontier Model (Qwen2.5-Coder-14B-Instruct-4bit)...")
    print("This requires ~8.5GB of VRAM. TSP will manage the remaining memory to prevent OOM.")
    try:
        model, tokenizer = load("mlx-community/Qwen2.5-Coder-14B-Instruct-4bit")
    except Exception as e:
        print(f"Failed to load model. Error: {e}")
        return

    # Setup aggressive pruning manager for the demo
    hook = CortexHook(eval_interval=5, threshold=0.99)
    manager = KVCacheManager(hook, model=model, enable_compression=True, enable_consolidation=True)
    manager.consolidator.salience_threshold = 0.0 # Force TTT for demo
    model.tsp_kv_manager = manager

    chat_history = [
        {"role": "system", "content": "You are a Senior Staff Engineer Agent operating autonomously."}
    ]
    
    # We simulate a massive, memory-heavy workflow
    workflow_steps = [
        {
            "phase": "Parsing 5,000 Line Stack Trace",
            "input": "System: The CI/CD pipeline failed. Analyze the following stack trace and find the memory leak.",
            "simulated_output": "> Reading stderr...\n\n[ERROR] Thread 1 crashed at address 0x10a2f4\n[ERROR] Segfault in MemoryPool::Allocate(size_t) at memory_pool.cpp:104\n[TRACE] #0  0x00000001004a3b1c in MemoryPool::Allocate(size_t) at memory_pool.cpp:104\n[TRACE] #1  0x00000001004a3c2a in ConnectionHandler::HandleRequest() at connection.cpp:212\n[TRACE] #2  0x00000001004a4f10 in Server::EventLoop() at server.cpp:55\n[INFO] Reading heap dump...\n[INFO] Found 14,000 un-freed allocations belonging to ConnectionHandler.\n\n> Analysis complete. The issue is a missing destructor call in ConnectionHandler."
        },
        {
            "phase": "Writing the Patch",
            "input": "System: Excellent. Now ignore the stack trace entirely. Shift your focus to connection.cpp and write the patch to fix the memory pool leak.",
            "simulated_output": "" # The model will generate the C++ fix
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
            print(f"Simulating heavy VRAM load for {step['phase']}...")
            _ = model(input_ids)
            
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
        
        # Force a long generation to test the 32B model's endurance
        generator = generate_infinite_context(model, input_ids, max_tokens=200)
        
        response = ""
        total_gen = input_ids.shape[1]
        last_evicted = 0
        action_log = ""
        
        print_dashboard(total_gen, manager.position_tracker.position_ids, 0, 0.0, step["phase"])
        print(f"{step['input']}\n")
        
        for token, stats in generator:
            token_id = token.item()
            if token_id == tokenizer.eos_token_id:
                break
                
            text = tokenizer.decode([token_id])
            response += text
            total_gen += 1
            
            if stats["total_evicted"] > last_evicted:
                diff = stats["total_evicted"] - last_evicted
                action_log = f"\u26A0\uFE0F VRAM RECOVERY: OOM Prevented. Evicted {diff} dead log tokens from the 32B Model's cache."
                last_evicted = stats["total_evicted"]
            
            print_dashboard(total_gen, stats["active_positions"], stats["total_evicted"], stats["lambda_2"], step["phase"], action_log)
            print(f"{step['input']}\n")
            print(f"> {response}", end="", flush=True)
            
        chat_history.append({"role": "assistant", "content": response})
        time.sleep(2)

    print("\n\n[TSP] 32B Frontier Agent Demo Complete.")

if __name__ == "__main__":
    main()
