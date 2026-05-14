import mlx.core as mx
from mlx_lm import load
from inference import generate_infinite_context
import time
import sys

def run():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    print(f"Loading {model_name}...")
    model, tokenizer = load(model_name)

    prompt = "Tell me a long, detailed story about a brave knight who climbs a mountain to fight a dragon, but discovers the dragon just wants to bake cookies."
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize
    input_ids = mx.array(tokenizer.encode(prompt_text))[None] # Add batch dim

    print(f"Input Prompt length: {input_ids.shape[1]} tokens")

    # Generate
    generator = generate_infinite_context(model, input_ids, max_tokens=100)
    
    print("\n--- GENERATION ---")
    
    for token_id in generator:
        token = token_id.item()
        text = tokenizer.decode([token])
        print(text, end="", flush=True)
        if token == tokenizer.eos_token_id:
            break
            
    print("\n\n--- DONE ---")

if __name__ == "__main__":
    run()
