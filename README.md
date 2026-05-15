# τ-Spectral Pruner (TSP) for MLX

**KV Cache Manager for MLX using Spectral Graph Theory**

TSP manages Large Language Model (LLM) KV caches by using spectral graph theory to identify and evict isolated token clusters. It maintains a bounded VRAM footprint by keeping a topologically connected context.

---

> ### 👦 **Explain Like I'm 6**
> Imagine an AI is like a student taking a really, really long test. To answer the questions, the student has to keep all the information they've read inside a tiny backpack (their memory). 
> 
> Normally, when the backpack gets full, the student either has to stop taking the test, or they have to throw away the *oldest* notes—even if those notes contain the most important instructions!
> 
> **TSP is a smart organizer for the backpack.** Instead of throwing away the oldest notes, it looks at everything in the bag and figures out which notes are completely unrelated to what the student is thinking about *right now*. It throws away the useless distraction notes (like a random math formula during a history essay) so the student never runs out of room and never forgets the important instructions.

---

## Architecture & Dependencies

TSP is designed for seamless integration with the Python `mlx-lm` ecosystem. It avoids serialization overhead and computation graph breaks by operating directly within the target execution environment.

**Core Dependency: $\tau$-Gate**
The system is fundamentally dependent on [**$\tau$-Gate**](https://github.com/steph4n-gh/tau-gate), a high-performance, zero-dependency Rust library that serves as the mathematical core for calculating spectral bisections on attention graphs.

*   **Zero-Latency FFI:** Python utilizes `ctypes` to bridge directly to the compiled `tau-gate` Rust engine (`libtau_gate.dylib`), calculating Fiedler vectors without inter-process communication (IPC) overhead.
*   **Graph Preservation:** Features like Test-Time Training (TTT) and Variance Compression execute within the Python MLX computation graph. This preserves lazy evaluation and enables automatic differentiation for real-time model updates.

## Perfect Memory via Holographic Paging

Users seeking absolute, verbatim recall ("perfect memory") often bolt on external Vector Databases (RAG). However, RAG introduces severe latency, requires embedding models, and injects retrieved text out of its original structural context, violating TSP's core mandate as a lightweight, zero-latency VRAM manager.

Instead, TSP implements **Holographic Paging**, exploiting Apple Silicon's Unified Memory architecture:
1. When a block of context is evicted, it is compressed into a single, dense **Macro-Token** via variance weighting.
2. The thousands of raw KV tensors are paged out of the active graph and parked in background system RAM.
3. The LLM retains the Macro-Token in its active context. If the model's attention heavily activates upon this semantic anchor, TSP intercepts the spike, pauses generation, and instantly pages the exact, mathematically perfect raw KV tensors back into the active GPU cache. 

## Neuro-Somatic Security & Adversarial Defense

A system that alters its own neural pathways at runtime introduces unique vulnerabilities. TSP implements three critical safeguards:
1.  **Read-Only Context Sandboxing:** If an evicted Thought Island contains untrusted tokens, Test-Time Training (LoRA gradient updates) is mathematically aborted, preventing adversarial prompt injections from permanently poisoning the model's weights.
2.  **Cryptographic Position Salting:** The Variance Compressor multiplies incoming token tensors by a cryptographically secure, session-specific random salt, making the semantic anchor's signature un-spoofable.
3.  **The Semantic Firewall (Joint Attention Bounding):** By monitoring the attention matrix for simultaneous spikes on predefined "Threat Sinks" and "Execution Sinks", TSP issues a `FATAL_BLOCK` to halt inference before destructive commands are synthesized.

## Use Cases for Persistent Agents

TSP is designed for scenarios where LLMs run continuously and accumulate unbounded context, such as CLI assistants and IDE integrations:

*   **Terminal Output Filtering:** When a CLI agent reads thousands of lines of logs (e.g., from `npm install` or test suites), TSP identifies the logs as an isolated block once the agent moves to the next task and prunes them from the active cache, retaining only the relevant context.
*   **Context-Switching in IDEs:** When switching between unrelated files or tasks, the semantic manifold shifts. TSP evicts the previous task's isolated tokens while preserving protected "sink" tokens (such as global project instructions and system prompts).
*   **Reducing Context Drag:** Continuously pruning irrelevant snippets prevents the context window from growing indefinitely, keeping inference times and token costs stable during long sessions.
*   **Instant Recovery (Checkpointing):** TSP can serialize its highly-compressed KV cache and topological map to disk (`safetensors` + `json`). This allows autonomous agents to survive process restarts and instantly resume their context without spending minutes re-processing days of log files.

## Security & Supply Chain

*   **Zero-Dependency Rust Core:** The mathematical library (`tau-gate`) is built without external dependencies (no crates) to minimize supply-chain risks.
*   **Local Processing:** Pruning and eigenvalue decomposition execute locally.
*   **Heuristic Nature:** TSP's bisection is a topological heuristic, not a guaranteed semantic filter. It assumes tokens separated by a maximum spectral gap are irrelevant.
*   **Sink Protection:** TSP allows developers to protect specific token ranges (sinks) from eviction.

## Quickstart (The DX Wrapper)

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

## Installation

```bash
git clone https://github.com/steph4n-gh/tsp-mlx
cd tsp-mlx/cpp
mkdir build && cd build
cmake ..
make -j4
```

## Documentation
*   **[What is TSP? (The Definitive Summary)](./docs/product_summary.md)**: An explanation of what the product is, what it does, and who it is for, scaled from a 6-year-old's understanding up to expert-level technical reality.
*   **[Theoretical Foundation: Nested Learning & TSP](./docs/nested_learning_analysis.md)**: An analysis connecting the TSP architecture to Google Research's *Nested Learning* paradigm.
*   **[Deep Dive: Configuration, Context, and Multi-Turn Survival](./docs/deep_dive.md)**: A technical, no-nonsense guide on how TSP integrates with LLMs, how it mathematically curates context, and why aggressive pruning doesn't break multi-turn chat templates.

## License
MIT License.
