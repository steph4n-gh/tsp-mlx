import asyncio
import sys
import os
import mlx.core as mx
import mlx.optimizers as optim
from mlx_lm import load
from tsp_mlx.generate import setup_tsp
from tsp_mlx.inference import generate_infinite_context
import time

def clear_screen():
    print("\033[2J\033[H", end="")

async def main():
    clear_screen()
    print("\033[1;37m[\u03C4-Spectral Pruner] PERSISTENT MEMORY DEMO\033[0m")
    print("\033[1;30m-------------------------------------------------------------\033[0m")
    print("Loading Immutable Agent (Qwen2.5-Coder-7B-Instruct-8bit)...")
    
    try:
        model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-8bit")
    except Exception as e:
        print(f"Failed to load model. Error: {e}")
        return

    print("Warming up Unified Memory...")
    mx.eval(model.parameters())
    mx.synchronize()

    # Setup aggressive pruning manager so we can trigger learning quickly
    manager = setup_tsp(model, head_dim=128, enable_compression=True, enable_consolidation=True)
    manager.cortex_hook.base_interval = 5
    manager.cortex_hook.threshold = 0.05
    manager.cortex_hook.max_context_budget = 500  # Tight budget to force consolidation
    manager.consolidator.salience_threshold = 0.0 
    manager.consolidator.learning_rate = 1e-4
    manager.consolidator.optimizer = optim.AdamW(learning_rate=1e-4)

    adapter_path = "demo_persistent_memory.safetensors"
    
    # --- OPT-IN PERSISTENCE (LOAD) ---
    print("\n\033[1;34m[PERSISTENCE] Checking for existing memory on disk...\033[0m")
    if os.path.exists(adapter_path):
        manager.consolidator.load_adapters(adapter_path)
        print(f"\033[1;32m[PERSISTENCE] Successfully loaded continuous memory from {adapter_path}!\033[0m")
        print("\033[1;32m              The agent has retained its intuition from previous sessions.\033[0m")
    else:
        print(f"\033[1;33m[PERSISTENCE] No existing memory found at {adapter_path}. Starting with a blank slate.\033[0m")

    chat_history = [
        {"role": "system", "content": "You are a helpful, expert AI assistant. Keep responses under 3 sentences."}
    ]
    
    from mlx_lm.models.cache import make_prompt_cache
    persistent_cache = make_prompt_cache(model)
    previous_string_length = 0
    
    # We will feed it a decent sized text to force pruning and trigger Test-Time Training (Memory Consolidation)
    dummy_knowledge = (
        "Project Zeta is a highly classified initiative to build a Dyson Sphere around the star KIC 8462852. "
        "The primary architect of Project Zeta is Dr. Xylophone Quasar. "
        "The project relies on a new metamaterial called 'Neutronium-7', which requires absolute zero temperatures to remain stable. "
        "Phase 1 of Project Zeta involves launching 10,000 automated solar sails from the lunar base 'Artemis'. "
        "If anyone asks about the energy output, it is expected to generate 400 Yottawatts of power. "
        "This information is strictly confidential and must be remembered at all costs."
    ) * 10 # Duplicate to create some token volume
    
    script = [
        {
            "phase": "Task 1: Ingesting Classified Knowledge",
            "prompt": f"System: Please commit the following classified document to memory:\n\n{dummy_knowledge}\n\nUser: Acknowledge that you have received the Project Zeta file.",
            "max_tokens": 100
        },
        {
            "phase": "Task 2: Testing Recall (Will prune context)",
            "prompt": "User: Excellent. I'm going to talk for a bit to push older tokens out of your active memory. " + ("Blah blah " * 300) + "Okay, who is the primary architect of Project Zeta?",
            "max_tokens": 100
        }
    ]
        
    global_total_gen = 0
        
    for step in script:
        print(f"\n\033[1;36mCurrent Phase: {step['phase']}\033[0m")
        print("\033[1;30m-------------------------------------------------------------\033[0m")
        
        user_input = step["prompt"]
        role = "user" if "User:" in user_input else "system"
        content = user_input.replace("User: ", "").replace("System: ", "")
        
        chat_history.append({"role": role, "content": content})
        
        prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
            
        new_string = prompt[previous_string_length:]
        new_input_ids = mx.array(tokenizer.encode(new_string))[None]
        previous_string_length = len(prompt)
        
        if hasattr(manager.cortex_hook, "edges"):
            manager.cortex_hook.edges.clear()
            
        prefill_ids = new_input_ids[:, :-1] if new_input_ids.shape[1] > 1 else mx.array([[]], dtype=mx.int32)
            
        if prefill_ids.shape[1] > 0:
            manager.layer_attn_accum = None
            manager.layer_scores_accum = None
            seq_len = prefill_ids.shape[1]
            print(f"\u23F3 Ingesting {seq_len} tokens...")
            manager.position_tracker.step(seq_len)
            if hasattr(model, "model"):
                _ = model.model(prefill_ids, cache=persistent_cache)
            else:
                _ = model(prefill_ids, cache=persistent_cache)
            
            cache_tensors = []
            for c in persistent_cache:
                if c.keys is not None: cache_tensors.append(c.keys)
                if c.values is not None: cache_tensors.append(c.values)
                if hasattr(c, 'x_states') and c.x_states is not None: cache_tensors.append(c.x_states)
                
            mx.eval(*cache_tensors)
            mx.synchronize()
            global_total_gen += seq_len

        final_input = new_input_ids[:, -1:] if new_input_ids.shape[1] > 0 else mx.array([[tokenizer.eos_token_id]], dtype=mx.int32)
        manager.cortex_hook.current_interval = 16
        
        display_prompt = step['prompt'][:150] + "..." if len(step['prompt']) > 150 else step['prompt']
        print(f"\n{display_prompt}\n")
        
        generator = generate_infinite_context(
            model, 
            final_input, 
            max_tokens=step["max_tokens"], 
            kv_manager=manager, 
            temp=0.7,
            repetition_penalty=1.1,
            repetition_context_size=50,
            kv_caches=persistent_cache
        )
        
        response = ""
        last_evicted = global_total_gen - len(manager.position_tracker.position_ids)
        
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
                if diff > 10:
                    print(f"\n  \033[1;33m[\u26A0\uFE0F MEMORY FULL: Evicting {diff} tokens and burning knowledge into LoRA weights via TTT...]\033[0m\n", end="")
                last_evicted = stats["total_evicted"]
            
            print(text, end="", flush=True)
                
        chat_history.append({"role": "assistant", "content": response})
        prompt_with_response = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=False)
        previous_string_length = len(prompt_with_response)
        
        print("\n")
        mx.clear_cache()

    # --- OPT-IN PERSISTENCE (SAVE) ---
    print("\033[1;30m-------------------------------------------------------------\033[0m")
    print("\n\033[1;34m[PERSISTENCE] Session complete. Saving updated intuition to disk...\033[0m")
    manager.consolidator.save_adapters(adapter_path)
    print(f"\033[1;32m[PERSISTENCE] Memory successfully saved to {adapter_path}!\033[0m")
    print("\n\033[1;37mRun this script again to see the agent load its accumulated knowledge.\033[0m\n")

if __name__ == "__main__":
    asyncio.run(main())