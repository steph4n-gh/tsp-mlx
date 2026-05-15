import mlx.core as mx
from mlx_lm import load
import asyncio
import time
import sys
from tsp_mlx.generate import setup_tsp
from tsp_mlx.inference import generate_infinite_context

def print_dashboard(doc_num, active_tokens, evicted_tokens, macro_tokens):
    sys.stdout.write('\033[2J\033[H') # Clear screen
    print(f"\033[1;36m[τ-Spectral Pruner] DEEP RESEARCHER AGENT\033[0m")
    print(f"Executing on: Qwen2.5-Coder-7B-Instruct")
    print(f"Currently Analyzing: Document {doc_num} of 10")
    print("-" * 60)
    print("\033[1;33mREAL-TIME VRAM METRICS:\033[0m")
    print(f"  ▶ Active Tokens In VRAM:  {active_tokens}")
    print(f"  ▶ Dead Code Evicted:      {evicted_tokens} tokens saved")
    print(f"  ▶ Topological Pages:      {macro_tokens} Macro-Tokens parked in RAM")
    print("-" * 60)
    print("\033[1;35mAGENT OUTPUT:\033[0m")

async def main():
    print("Booting Deep Researcher Agent...")
    model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-8bit")
    
    # 1. Setup TSP with extremely aggressive pruning to force the demo
    manager = setup_tsp(model, head_dim=128, enable_compression=True, enable_consolidation=True)
    manager.cortex_hook.base_interval = 16
    manager.cortex_hook.threshold = 0.8
    manager.consolidator.salience_threshold = 0.2 # TTT only for salient items
    
    chat_history = [
        {"role": "system", "content": "You are a Senior Cryptography Analyst. Your job is to read dense technical specifications and summarize their core vulnerabilities in exactly one sentence."}
    ]
    
    # Dense, complex simulated research blocks
    research_blocks = [
        "RFC Draft 8990: The proposed Quantum-Resistant Key Encapsulation Mechanism relies on learning with errors (LWE) over module lattices. However, the parameter set chosen (n=512, q=3329) presents a dangerously tight margin against primal attacks using BKZ-2.0 lattice reduction if the attacker has access to a sufficiently large quantum oracle.",
        "System Architecture V4.2: The unified memory architecture utilizes a single physical bus for both CPU instruction fetching and GPU texture mapping. While this reduces PCIe bottlenecks, it introduces a theoretical side-channel timing vulnerability. If a malicious shader precisely times the memory access latency during a parallel CPU encryption routine, it could theoretically infer the AES round keys.",
        "Zero-Knowledge Rollup Spec: The zk-SNARK implementation uses the Groth16 proof system. While efficient in verification, the trusted setup phase relies on a Multi-Party Computation (MPC) ceremony. If the ceremony's toxic waste is not verifiably destroyed by at least one honest participant, the entire proving system can be universally forged.",
        "Consensus Protocol Analysis: The Byzantine Fault Tolerant (BFT) consensus mechanism requires a 2/3 supermajority for finality. However, the gossip protocol's timeout parameter (t_delta = 50ms) is dangerously low for global network latency. This could lead to a liveness failure (chain halt) during periods of high BGP routing instability.",
        "Smart Contract Audit: The liquidity pool contract implements a constant-product AMM (x*y=k). The `flashLoan` function allows uncollateralized borrowing provided the balance is restored by the end of the transaction. However, the fee calculation relies on a state variable `totalReserves` that is updated *before* the loan is returned, opening the door to a reentrancy attack.",
        "Network Protocol IEEE 802.11: The BGP routing tables are updated dynamically based on AS-PATH lengths. However, the lack of cryptographic signing on route advertisements (RPKI adoption is low) means an attacker can broadcast a /24 prefix claim with a shorter path, successfully hijacking traffic destined for critical infrastructure.",
        "Database Architecture: The distributed SQL database uses a Paxos variant for leader election. To optimize write latency, the `fsync` system call is batched every 500ms instead of per-transaction. During a hard power failure of the primary leader, the system guarantees a minimum of 500ms of committed transaction loss, violating strict ACID durability.",
        "Operating System Kernel: The memory management unit (MMU) uses a translation lookaside buffer (TLB) for fast virtual-to-physical address mapping. The OS kernel flushes the TLB lazily during context switches to improve performance. This lazy flushing theoretically allows a user-space process to access stale memory pages belonging to a previously executed, highly-privileged process.",
        "Authentication Protocol: The OAuth 2.0 implementation utilizes implicit grant flow for single-page applications. The access token is returned directly in the URI fragment. Because URI fragments are often logged by browser extensions or visible in referer headers, the token is highly susceptible to interception and replay attacks.",
        "Hardware Security Module (HSM): The secure enclave stores private RSA keys in tamper-resistant NVRAM. However, the physical power supply line lacks a low-pass filter. An attacker with physical access could induce micro-voltage drops (glitching) during the modular exponentiation phase, causing a calculation error that leaks the private key via the Chinese Remainder Theorem."
    ]

    for i, block in enumerate(research_blocks, 1):
        chat_history.append({"role": "user", "content": f"Analyze Document {i}:\n{block}\n\nWhat is the core vulnerability? Reply in ONE sentence."})
        
        prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
        input_ids = mx.array(tokenizer.encode(prompt))[None]
        
        # We use generate_infinite_context directly to utilize the shared manager
        generator = generate_infinite_context(
            model, 
            input_ids, 
            max_tokens=100, 
            kv_manager=manager, 
            temp=0.7, 
            repetition_penalty=1.15,
            repetition_context_size=50
        )
        
        response_text = ""
        token_count = 0
        async for token, stats in generator:
            token_id = token.item()
            if token_id == tokenizer.eos_token_id:
                break
            
            chunk = tokenizer.decode([token_id])
            response_text += chunk
            token_count += 1
            
            # Print dashboard every 5 tokens to reduce flicker but show progress
            if token_count % 5 == 0:
                print_dashboard(i, len(stats.get("active_positions", [])), stats.get("total_evicted", 0), len(manager.topological_pages))
                print(f"> {response_text}")

        # Final dashboard update for this document
        print_dashboard(i, len(stats.get("active_positions", [])), stats.get("total_evicted", 0), len(manager.topological_pages))
        print(f"> {response_text}")
        
        chat_history.append({"role": "assistant", "content": response_text})
        
        # Rigorous GC after each document
        mx.clear_cache()
        import gc
        gc.collect()
        
        time.sleep(2) # Pause so user can read the output

    print("\n\n\033[1;32m[SUCCESS] Infinite Horizon Research Loop Complete.\033[0m")
    print(f"Final VRAM State: {len(stats.get('active_positions', []))} active tokens. (Model avoided OOM over {len(research_blocks)} complex documents via {len(manager.topological_pages)} Topological Pages!)")

if __name__ == "__main__":
    asyncio.run(main())
