import sys
import mlx.core as mx
from mlx_lm import load
from tsp_mlx import generate_infinite_context
from tsp_mlx.cortex_hook import CortexHook
import time

def clear_screen():
    print("\033[2J\033[H", end="")

def print_dashboard(total_gen, active_ids, evicted, lambda2, turn, total_turns):
    clear_screen()
    print(f"\033[1;37m[\u03C4-Spectral Pruner] AUTOMATED EXECUTIVE FUNCTION SHOWCASE (Turn {turn}/{total_turns})\033[0m")
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
        print(f"Failed to load model. Error: {e}")
        return

    chat_history = [
        {"role": "system", "content": "You are a helpful, very brief AI assistant. Always keep your answers under 3 sentences."}
    ]
    
    # Scripted, rambling conversation to force semantic shifts
    script = [
        "Hi! Let's start a long conversation. Tell me a 3 sentence story about a brave knight named Sir Arthur.",
        "Interesting. Now, switch topics completely. Explain the process of photosynthesis in 3 sentences.",
        "Back to Sir Arthur. What was the name of his horse, and what color was it?",
        "Okay, can you write a python script to calculate the Fibonacci sequence? Just the code, please.",
        "What are the main differences between classical mechanics and quantum mechanics? Keep it brief.",
        "Sir Arthur is riding his horse. What obstacle does he encounter on the road?",
        "Name 5 different types of clouds and their characteristics.",
        "How did Sir Arthur overcome the obstacle?",
        "What's the recipe for a classic Margherita pizza?",
        "Sir Arthur finds a treasure chest. What is inside it?",
        "Explain the concept of inflation in economics.",
        "Who is the author of '1984', and what is the book's main theme?",
        "The treasure inside the chest is glowing. What happens when Sir Arthur touches it?",
        "List the planets in our solar system in order from the sun.",
        "What are the rules of Chess regarding the movement of the Knight piece?",
        "Sir Arthur is transported to a new realm. Describe this realm in 3 sentences.",
        "How do vaccines work to protect the immune system?",
        "What are the three laws of motion by Isaac Newton?",
        "Who does Sir Arthur meet in this new realm?",
        "Translate 'Hello, how are you doing today?' into Spanish and French.",
        "Sir Arthur's new friend gives him a quest. What is the quest?",
        "Explain the difference between a virus and a bacteria.",
        "What are the primary colors in painting?",
        "Sir Arthur begins his quest. What is his first step?",
        "Summarize the plot of Romeo and Juliet.",
        "How does a four-stroke engine work?",
        "Sir Arthur faces a dragon. How does he defeat it without killing it?",
        "What is the capital of Australia?",
        "Finally, summarize Sir Arthur's entire journey from the beginning of our chat."
    ]
    
    for turn, user_input in enumerate(script, 1):
        chat_history.append({"role": "user", "content": user_input})
        
        prompt = tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=True
        )
        
        input_ids = mx.array(tokenizer.encode(prompt))[None]
        
        generator = generate_infinite_context(model, input_ids, max_tokens=100)
        
        response = ""
        total_gen = input_ids.shape[1]
        
        print_dashboard(total_gen, list(range(total_gen)), 0, 0.0, turn, len(script))
        print(f"\033[1;34mUser: {user_input}\033[0m\n")
        
        for token, stats in generator:
            token_id = token.item()
            if token_id == tokenizer.eos_token_id:
                break
                
            text = tokenizer.decode([token_id])
            response += text
            total_gen += 1
            
            print_dashboard(total_gen, stats["active_positions"], stats["total_evicted"], stats["lambda_2"], turn, len(script))
            print(f"\033[1;34mUser: {user_input}\033[0m\n")
            print(f"> {response}", end="", flush=True)
            
        chat_history.append({"role": "assistant", "content": response})
        time.sleep(1) # Pause so the user can read the final output of the turn

    print("\n\n[TSP] Automated Demo Complete.")

if __name__ == "__main__":
    main()
