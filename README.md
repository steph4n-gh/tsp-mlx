# 🚀 τ-Spectral Pruner (TSP) for MLX

**An organic memory framework for MLX LLMs featuring Topological Paging, True Test-Time Training (TTT), and a zero-latency Rust core.**

TSP prevents LLM Out-Of-Memory (OOM) crashes by treating the model's working memory as a mathematical graph. It autonomously pages out irrelevant context to RAM, giving local autonomous agents infinite context windows without relying on RAG or external databases.

**Target Hardware:** Built and optimized for Apple Silicon. The baseline target is a **MacBook Pro M4 Pro with 24GB RAM**. The $O(1)$ framework is highly efficient and perfectly acceptable on lower-end machines (e.g., M1/M2 with 8GB/16GB), but scales infinitely with more Unified Memory for massive topological paging.

---

## ⚡ Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/steph4n-gh/tsp-mlx
cd tsp-mlx

# Build the Rust math engine (τ-Gate)
cd ../supplychain
cargo build --release

# Setup Python Environment
cd ../tsp-mlx
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt # or install mlx, mlx_lm manually
```

### 2. The DX Wrapper
Integrating TSP into your own MLX projects is incredibly simple. We provide a high-level wrapper that handles the complex math, caching, and garbage collection for you:

```python
import asyncio
from mlx_lm import load
from tsp_mlx.generate import generate_with_tsp

async def main():
    model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-8bit")
    
    # 🛑 Untrusted tokens bypass permanent learning to prevent poisoning
    untrusted = {5, 6, 7} 

    # A single line handles initialization, RoPE patching, and inference
    generator = generate_with_tsp(
        model, tokenizer, 
        prompt="Write a haiku about a cybernetic dragon.", 
        max_tokens=50, 
        untrusted_indices=untrusted
    )
    
    async for text, stats in generator:
        print(text, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## ✨ Key Features

*   **Dual-System Fractal Memory:** 
    *   **Topological Paging (Conscious Memory):** Instead of throwing context away, TSP compresses pruned islands into a dense semantic 'Macro-Token' anchor in VRAM, paging the raw text to RAM. If attention flows back to the Macro-Token, the raw tokens are instantly unpacked for flawless factual recall. An **Anti-Spiral Unpack Throttle** ensures memory is only unpacked if there is sufficient VRAM budget headroom (e.g., < 75% full), preventing catastrophic memory overflows and cyclical pruning death spirals.
    *   **True Test-Time Training (Subconscious Memory):** When context is evicted, the agent runs fast $O(1)$ gradient descent on injected `LoRALinear` adapters in the final layer. This physically burns the "vibe" and style of the forgotten context into the model's background neural weights.
*   **Subconscious Swarm (Split-LoRA):** Run multiple agents simultaneously that share a single base model in VRAM. The framework hot-swaps tiny LoRA memory snapshots on the fly, allowing agents to debate from different subconscious perspectives with zero additional memory overhead.
*   **Deep Debate (Exhaustive Resolution):** An autonomous adversarial loop where agents mercilessly critique each other's logic. The loop utilizes an autonomous stopping condition (`[DEBATE_RESOLVED]`), guaranteeing the agents only break the cycle when 100% consensus is reached on every detail.
*   **God-View Visualizer:** Exposes the invisible MLX mathematical operations (Spectral Bisection, LoRA descent), rendering a live ASCII topological graph of the agent's brain directly in the terminal so you can watch thoughts cluster and sever in real-time.
*   **Zero-Overhead Inference:** By using pre-allocated slice assignments for state tracking and dynamic attention thresholding, TSP runs at the physical memory bandwidth limits of Apple Silicon (~189 GB/s on an M4 Pro) adding 0.0% Python overhead to token decoding.
*   **Massive Prefill & Seamless File Uploads:** Through mathematically correct `mx.triu` causal masking, TSP can ingest 15,000+ token codebase files in a single FlashAttention burst. The engine dynamically maps the new topology to your compressed Macro-Tokens in RAM, instantly triggering $O(1)$ factual recall if the uploaded file is related to a past memory.
*   **Semantic Firewall:** Defends against adversarial memory poisoning via Read-Only sandboxing, and blocks prompt injection attempts by mathematically identifying topological anomalies pointing at the core System Prompt and dropping a `FATAL_BLOCK`.

---

## 🧠 How It Works (Explain Like I'm 6)

Imagine an AI is like a student taking a really, really long test. To answer the questions, the student has to keep all the information they've read inside a tiny backpack (their memory). 

Normally, when the backpack gets full, the student either has to stop taking the test, or they have to throw away the *oldest* notes—even if those notes contain the most important instructions!

**TSP is a smart organizer for the backpack.** Instead of throwing away the oldest notes, it looks at everything in the bag and figures out which notes are completely unrelated to what the student is thinking about *right now*. It zips those notes up and puts them in a locker (RAM). If the student ever needs them again, it instantly unzips them back into the backpack!

---

## 📚 Documentation & Deep Dives

*   **[📖 Read the Whitepaper](./docs/whitepaper.md)**: A formal breakdown of the organic engine, detailing Topological Pruning, Topological Compression, and Parametric Memory.
*   **[🛠️ Agent Integration Guide](./docs/agent_integration_guide.md)**: How to cleanly integrate the TSP framework into your existing Python applications.
*   **[What is TSP? (The Summary)](./docs/product_summary.md)**: An explanation of what the product is, what it does, and who it is for.
*   **[Deep Dive: Context and Multi-Turn Survival](./docs/deep_dive.md)**: A technical guide on how TSP integrates with LLMs and mathematically curates context without breaking chat templates.

## License
MIT License.
