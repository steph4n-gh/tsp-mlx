# 🚀 τ-Spectral Pruner (TSP) for MLX

**An organic memory framework for MLX LLMs featuring Topological Pruning, Holographic Paging, and Test-Time Training via a zero-latency Rust core.**

TSP prevents LLM Out-Of-Memory (OOM) crashes by treating the model's working memory as a mathematical graph. It autonomously pages out irrelevant context to RAM, giving local autonomous agents infinite context windows without relying on RAG or external databases.

---

## ⚡ Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/steph4n-gh/tsp-mlx
cd tsp-mlx

# Build the Rust math engine
cd cpp
mkdir build && cd build
cmake ..
make -j4
```

### 2. The DX Wrapper
Integrating TSP into your own MLX projects is incredibly simple. We provide a high-level wrapper that handles the complex math, caching, and garbage collection for you:

```python
import asyncio
from mlx_lm import load
from tsp_mlx.generate import generate_with_tsp

async def main():
    model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    
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

*   **Holographic Paging:** Instead of RAG, TSP compresses evicted context into dense Semantic Macro-Tokens via variance weighting, while parking the raw KV tensors in RAM. If the LLM's attention spikes on a Macro-Token, the exact raw tensors are instantly swapped back into the active GPU cache.
*   **Permanent Learning (TTT):** When context is evicted, the agent runs Test-Time Training (gradient descent) on injected LoRA adapters. These weights are serialized to `.safetensors`, giving the agent a permanent "muscle memory" of what it read across server restarts.
*   **Neuro-Somatic Security:** Defends against adversarial memory poisoning via Read-Only sandboxing, Cryptographic Position Salting to prevent holographic spoofing, and a Semantic Firewall that blocks dangerous commands at the mathematical intent layer.

---

## 🧠 How It Works (Explain Like I'm 6)

Imagine an AI is like a student taking a really, really long test. To answer the questions, the student has to keep all the information they've read inside a tiny backpack (their memory). 

Normally, when the backpack gets full, the student either has to stop taking the test, or they have to throw away the *oldest* notes—even if those notes contain the most important instructions!

**TSP is a smart organizer for the backpack.** Instead of throwing away the oldest notes, it looks at everything in the bag and figures out which notes are completely unrelated to what the student is thinking about *right now*. It throws away the useless distraction notes (like a random math formula during a history essay) so the student never runs out of room and never forgets the important instructions.

---

## 📚 Documentation & Deep Dives

*   **[📖 Read the Whitepaper](./docs/whitepaper.md)**: A formal breakdown of the organic engine, detailing Topological Pruning, Holographic Paging, and Parametric Memory.
*   **[🛠️ Agent Integration Guide](./docs/agent_integration_guide.md)**: How to cleanly integrate the TSP framework into your existing Python applications.
*   **[What is TSP? (The Summary)](./docs/product_summary.md)**: An explanation of what the product is, what it does, and who it is for.
*   **[Deep Dive: Context and Multi-Turn Survival](./docs/deep_dive.md)**: A technical guide on how TSP integrates with LLMs and mathematically curates context without breaking chat templates.

## License
MIT License.
