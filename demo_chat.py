import sys
import mlx.core as mx
from mlx_lm import load
from tsp_mlx import generate_infinite_context
from tsp_mlx.cortex_hook import CortexHook
import time

def clear_screen():
    print("\033[2J\033[H", end="")

def print_dashboard(total_gen, active_ids, evicted, lambda2):
    clear_screen()
    print("\033[1;37m[\u03C4-Spectral Pruner] THE EXECUTIVE FUNCTION SHOWCASE\033[0m")
    print("\033[1;30m-------------------------------------------------------------\033[0m")
    
    # Real-time Metrics
    print("\n\033[1;37mREAL-TIME METRICS:\033[0m")
    print(f"  \u25B6 \033[1mTotal Generation:\033[0m {total_gen} tokens")
    print(f"  \u25B6 \033[1mActive Context:\033[0m   {len(active_ids)} tokens")
    print(f"  \u25B6 \033[1mTotal Evicted:\033[0m    {evicted} tokens")
    print(f"  \u25B6 \033[1mSpectral Gap:\033[0m     {lambda2:.8f} (\u03BB\u2082)")
    
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

def main():
    print("Loading model (Qwen2.5-0.5B-Instruct)...")
    try:
        model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    except Exception as e:
        print(f"Failed to load model. Ensure mlx-lm is installed and you have internet access. Error: {e}")
        return

    # To trigger pruning more often for the demo, we could monkey-patch the hook's eval interval,
    # but let's just use the defaults defined in inference.py (eval_interval=16)

    chat_history = [
        {"role": "system", "content": "You are a helpful, very brief AI assistant. Always keep your answers under 3 sentences."}
    ]
    
    print("\nReady! Type your message (or 'quit' to exit):")
    
    while True:
        try:
            user_input = input("\n\033[1;34mUser: \033[0m")
            if user_input.lower() in ['quit', 'exit']:
                break
        except (KeyboardInterrupt, EOFError):
            break
            
        chat_history.append({"role": "user", "content": user_input})
        
        prompt = tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=True
        )
        
        input_ids = mx.array(tokenizer.encode(prompt))[None]
        
        # We need a way to track total tokens across turns. For this simple demo, 
        # generate_infinite_context resets the position tracker on every call. 
        # In a real app, the KV cache and tracker would persist across turns.
        # But even resetting it, we can see the pruning happen within a single long response.
        
        generator = generate_infinite_context(model, input_ids, max_tokens=200)
        
        response = ""
        total_gen = input_ids.shape[1]
        
        # Initial Dashboard
        print_dashboard(total_gen, list(range(total_gen)), 0, 0.0)
        
        for token, stats in generator:
            token_id = token.item()
            if token_id == tokenizer.eos_token_id:
                break
                
            text = tokenizer.decode([token_id])
            response += text
            total_gen += 1
            
            # Print the dashboard
            print_dashboard(total_gen, stats["active_positions"], stats["total_evicted"], stats["lambda_2"])
            print(f"> {response}", end="", flush=True)
            
        chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
