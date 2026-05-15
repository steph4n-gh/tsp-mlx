# τ-Spectral Pruner (TSP) for MLX

**C++ KV Cache Manager for Apple Silicon using Spectral Graph Theory**

TSP manages Large Language Model (LLM) KV caches by using spectral graph theory to identify and evict isolated token clusters. It maintains a bounded VRAM footprint by keeping a topologically connected context.

---

> ### 👦 **Explain Like I'm 6**
> Imagine an AI is like a student taking a really, really long test. To answer the questions, the student has to keep all the information they've read inside a tiny backpack (their memory). 
> 
> Normally, when the backpack gets full, the student either has to stop taking the test, or they have to throw away the *oldest* notes—even if those notes contain the most important instructions!
> 
> **TSP is a smart organizer for the backpack.** Instead of throwing away the oldest notes, it looks at everything in the bag and figures out which notes are completely unrelated to what the student is thinking about *right now*. It throws away the useless distraction notes (like a random math formula during a history essay) so the student never runs out of room and never forgets the important instructions.

---

## Dual-Stack Architecture & Dependencies

TSP utilizes a dual-stack architecture to ensure maximum performance and compatibility across different integration environments. The system avoids serialization overhead and computation graph breaks by operating directly within the target execution environment.

**Core Dependency: $\tau$-Gate**
Both stacks are fundamentally dependent on [**$\tau$-Gate**](https://github.com/steph4n-gh/tau-gate), a high-performance, zero-dependency Rust library that serves as the mathematical core for calculating spectral bisections on attention graphs.

### 1. The Python Stack (`mlx-lm` Integration)
Designed for seamless integration with the Python `mlx-lm` ecosystem.
*   **Zero-Latency FFI:** Python utilizes `ctypes` to bridge directly to the compiled `tau-gate` Rust engine (`libtau_gate.dylib`), calculating Fiedler vectors without inter-process communication (IPC) overhead.
*   **Graph Preservation:** Advanced features like Test-Time Training (TTT) and Topological Compression execute within the Python MLX computation graph. This preserves lazy evaluation and enables automatic differentiation for real-time model updates.

### 2. The Native C++ Stack
Designed for standalone, high-performance C++ inference engines using `mlx-cxx`.
*   **Native Execution:** Implements the `KVCacheManager`, TTT gradients, and SubManifold Autoencoders natively in C++.
*   **Static Linking:** Directly links against the compiled Rust library (`libtau_gate.a`) for native memory access and optimal topological analysis speed.

## Architectural Philosophy: Why Not RAG? (Perfect Memory vs. Intuition)

Users seeking absolute, verbatim recall ("perfect memory") could easily extend TSP by piping the evicted "Thought Islands" into a Vector Database for Retrieval-Augmented Generation (RAG). However, TSP deliberately omits RAG from its default architecture for several critical reasons:

1.  **Zero-Dependency & Low Latency:** TSP is engineered to operate at the foundational matrix level. Introducing a Vector DB requires external dependencies, embedding models, and disk I/O, which violate TSP's core mandate as a lightweight, zero-latency VRAM manager.
2.  **Intuition vs. Archival:** RAG acts as an external hard drive (L2 Cache), solving *data retrieval*. TSP focuses on the AI's *intuition and executive function*. Through Test-Time Training (TTT), TSP compresses concepts into the model's weights ("muscle memory"), prioritizing behavioral adaptation and style over verbatim text recall.
3.  **Context Bloat:** RAG achieves recall by injecting old text back into the active prompt, inherently bloating the KV cache. TSP is designed to explicitly keep the KV cache mathematically lean.

**Extending TSP:** Developers are encouraged to combine TSP with their own RAG pipelines for composite memory: use TSP to keep the active GPU context pure, let the `MemoryConsolidator` bake behavioral nuances into the model's weights (TTT), and route the evicted tokens to a database for permanent archival.

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
