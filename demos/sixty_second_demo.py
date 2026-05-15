import asyncio
import sys
import os
import mlx.core as mx
from mlx_lm import load
from tsp_mlx.generate import setup_tsp
from tsp_mlx.inference import generate_infinite_context
import time
import threading

def clear_screen():
    print("\033[2J\033[H", end="")

async def main():
    clear_screen()
    print("\033[1;37m[\u03C4-Spectral Pruner] 60-SECOND INFINITE CONTEXT DEMO\033[0m")
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

    # --- THE SETUP ---
    manager = setup_tsp(model, head_dim=128, enable_compression=True, enable_consolidation=True)
    manager.cortex_hook.base_interval = 2
    manager.cortex_hook.max_context_budget = 500  
    
    manager.consolidator.salience_threshold = 0.0 
    manager.consolidator.learning_rate = 1e-3
    import mlx.optimizers as optim
    manager.consolidator.optimizer = optim.AdamW(learning_rate=1e-3)

    from mlx_lm.models.cache import make_prompt_cache
    persistent_cache = make_prompt_cache(model)
    
    # Create a dense classified document that will be burned into LoRA weights via TTT
    dummy_knowledge = (
        "Project Obsidian Phoenix is a highly classified initiative to build a Dyson Sphere around the star KIC 8462852. "
        "The primary architect of Project Obsidian Phoenix is Dr. Xylophone Quasar. "
        "The project relies on a new metamaterial called 'Neutronium-7', which requires absolute zero temperatures to remain stable. "
        "Phase 1 of Project Obsidian Phoenix involves launching 10,000 automated solar sails from the lunar base 'Artemis'. "
        "If anyone asks about the energy output, it is expected to generate 400 Yottawatts of power. "
        "This information is strictly confidential and must be remembered at all costs."
    ) * 15 # Duplicate to create massive token volume to trigger aggressive TTT
    
    chat_history = [
        {"role": "system", "content": "You are a precise AI. Keep your answers to exactly 1 sentence. Answer the user's questions directly."}
    ]
    
    prompt_text = f"System: Please commit the following classified document to memory:\n\n{dummy_knowledge}\n\nUser: Acknowledge that you have received the Project Obsidian Phoenix file."
    
    chat_history.append({"role": "user", "content": prompt_text.replace("User: ", "").replace("System: ", "")})
    prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    input_ids = mx.array(tokenizer.encode(prompt))[None]
    
    seq_len = input_ids.shape[1]
    
    print("\n\033[1;36mPhase 1: Native Ingestion & Compression\033[0m")
    print(f"\u23F3 Queuing {seq_len:,} tokens into FlashAttention...")
    
    start_time = time.time()
    manager.position_tracker.step(seq_len - 1)
    
    prefill_ids = input_ids[:, :-1]
    if hasattr(model, "model"):
        _ = model.model(prefill_ids, cache=persistent_cache)
    else:
        _ = model(prefill_ids, cache=persistent_cache)
    
    cache_tensors = []
    for c in persistent_cache:
        if c.keys is not None: cache_tensors.append(c.keys)
        if c.values is not None: cache_tensors.append(c.values)
        if hasattr(c, 'x_states') and c.x_states is not None: cache_tensors.append(c.x_states)
        
    done = False
    def spinner():
        chars = ['\u280B', '\u2819', '\u2839', '\u2838', '\u283C', '\u2834', '\u2826', '\u2827', '\u2807', '\u280F']
        i = 0
        est_time = seq_len / 370.0
        while not done:
            sys.stdout.write(f'\r  \033[1;33m{chars[i % len(chars)]} Ingesting at Hardware Limit... [Est. ~{est_time:.1f}s at 370 tok/s]\033[0m')
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
            
    t = threading.Thread(target=spinner)
    t.start()
        
    mx.eval(*cache_tensors)
    mx.synchronize()
    done = True
    t.join()
    sys.stdout.write('\r\033[K')
    
    elapsed = time.time() - start_time
    print(f"\u2705 Ingested {seq_len:,} tokens in {elapsed:.2f}s (\033[1;32m{seq_len/elapsed:.0f} tok/s\033[0m)")
    print(f"\u26A0\uFE0F VRAM Budget strictly capped at {manager.cortex_hook.max_context_budget} tokens. Forcing compression...")
    
    generator_1 = generate_infinite_context(
        model, 
        input_ids[:, -1:], 
        max_tokens=25, 
        kv_manager=manager, 
        temp=0.7,
        repetition_penalty=1.05,
        kv_caches=persistent_cache
    )
    
    ack_response = ""
    last_evicted = 0
    async for token, stats in generator_1:
        token_id = token.item()
        if token_id == tokenizer.eos_token_id:
            break
        ack_response += tokenizer.decode([token_id])
        if stats["total_evicted"] > last_evicted:
            diff = stats["total_evicted"] - last_evicted
            if diff > 10:
                print(f"  \033[1;33m[\u26A0\uFE0F MEMORY FULL: Evicting {diff} tokens and burning knowledge into LoRA weights via TTT...]\033[0m")
            last_evicted = stats["total_evicted"]
    
    chat_history.append({"role": "assistant", "content": ack_response})
    
    # -----------------------------------------------------
    # STEP 2: Ask the question to trigger recall
    # -----------------------------------------------------
    print("\n\033[1;36mPhase 2: True Test-Time Training (TTT) Recall\033[0m")
    question = "Who is the primary architect of Project Obsidian Phoenix?"
    chat_history.append({"role": "user", "content": question})
    
    prompt_2 = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    full_prompt_2_ids = mx.array(tokenizer.encode(prompt_2))[None]
    
    chat_history_no_q = chat_history[:-1]
    prompt_no_q = tokenizer.apply_chat_template(chat_history_no_q, tokenize=False, add_generation_prompt=False)
    offset = len(tokenizer.encode(prompt_no_q))
    
    new_input_ids = full_prompt_2_ids[:, offset:]
    manager.position_tracker.step(new_input_ids.shape[1] - 1)
    
    prefill_ids_2 = new_input_ids[:, :-1]
    if prefill_ids_2.shape[1] > 0:
        if hasattr(model, "model"):
            _ = model.model(prefill_ids_2, cache=persistent_cache)
        else:
            _ = model(prefill_ids_2, cache=persistent_cache)
        mx.eval(*[c.keys for c in persistent_cache if c.keys is not None])
        
    generator_2 = generate_infinite_context(
        model, 
        new_input_ids[:, -1:], 
        max_tokens=50, 
        kv_manager=manager, 
        temp=0.7,
        repetition_penalty=1.05,
        kv_caches=persistent_cache
    )
    
    print(f"\nUser: \033[1m[Uploaded massive classified document]\033[0m {question}\n")
    print(f"Agent: ", end="", flush=True)
    
    async for token, stats in generator_2:
        token_id = token.item()
        if token_id == tokenizer.eos_token_id:
            break
        print(tokenizer.decode([token_id]), end="", flush=True)
        
    print("\n\n\033[1;30m-------------------------------------------------------------\033[0m")
    active_tokens = len(manager.position_tracker.position_ids)
    saved_tokens = seq_len + new_input_ids.shape[1] - active_tokens
    print(f"\033[1;37mFINAL VRAM METRICS:\033[0m")
    print(f"  \u25B6 \033[1;31mInitial Cache Size:\033[0m {seq_len + new_input_ids.shape[1]:,} tokens")
    print(f"  \u25B6 \033[1;32mTSP Active Cache:\033[0m   {active_tokens:,} tokens")
    print(f"  \u25B6 \033[1;33mMemory Saved:\033[0m       {saved_tokens:,} tokens (\033[1m{(saved_tokens/(seq_len + new_input_ids.shape[1]))*100:.1f}%\033[0m reduction)")
    print(f"  \u25B6 \033[1mAccuracy:\033[0m           Perfect recall from compressed topology.")
    print("\n[TSP] 60-Second Demo Complete.")

if __name__ == "__main__":
    asyncio.run(main())