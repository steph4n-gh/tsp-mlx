import mlx.core as mx
from mlx_lm import load
from tsp_mlx.generate import generate_with_tsp
import asyncio

async def main():
    print("\033[1;36m=== Testing the TSP Developer Experience (DX) Wrapper ===\033[0m")
    print("Loading model...")
    model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    
    prompt = "Write a haiku about a cybernetic dragon."
    print(f"\nPrompt: {prompt}\n")
    print("\033[1;33mGeneration Output:\033[0m")
    
    # We pass a set of untrusted_indices just to show the feature integrates easily
    untrusted = {5, 6, 7}
    
    # The single line that replaces 50 lines of complex setup!
    generator = generate_with_tsp(model, tokenizer, prompt, max_tokens=50, untrusted_indices=untrusted)
    
    stats_history = []
    async for text, stats in generator:
        print(text, end="", flush=True)
        stats_history.append(stats)
        
    print(f"\n\n\033[1;32m[SUCCESS] DX Wrapper works seamlessly!\033[0m")
    print(f"Final internal active positions: {len(stats_history[-1]['active_positions'])}")

if __name__ == "__main__":
    asyncio.run(main())
