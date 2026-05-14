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

## Architecture & Dependencies

TSP is built as a C++ library that integrates with the MLX framework.

**Core Dependency: $\tau$-Gate**
TSP is fundamentally dependent on [**$\tau$-Gate**](https://github.com/steph4n-gh/tau-gate), which serves as its mathematical core. 
1.  **MLX C++ Integration:** Directly interfaces with the MLX C++ API to manage KV cache tensors.
2.  **$\tau$-Gate Engine:** The `tau-gate` Rust static library is required to perform the spectral bisection on the attention graph.
3.  **FFI Bridge:** C++ communicates with the Rust engine via C FFI to avoid inter-process communication overhead.

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
