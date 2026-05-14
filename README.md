# τ-Spectral Pruner (TSP) for MLX

**C++ KV Cache Manager for Apple Silicon using Spectral Graph Theory**

TSP manages Large Language Model (LLM) KV caches by using spectral graph theory to identify and evict isolated token clusters. It maintains a bounded VRAM footprint by keeping a topologically connected context.

## Architecture

TSP is built as a C++ library that integrates with the MLX framework.

1.  **MLX C++ Integration:** Directly interfaces with the MLX C++ API to manage KV cache tensors.
2.  **$\tau$-Gate:** A Rust static library that performs spectral bisection on the attention graph.
3.  **FFI Bridge:** C++ communicates with the Rust engine via C FFI to avoid inter-process communication overhead.

## Use Cases for Persistent Agents

TSP is designed for scenarios where LLMs run continuously and accumulate unbounded context, such as CLI assistants and IDE integrations:

*   **Terminal Output Filtering:** When a CLI agent reads thousands of lines of logs (e.g., from `npm install` or test suites), TSP identifies the logs as an isolated block once the agent moves to the next task and prunes them from the active cache, retaining only the relevant context.
*   **Context-Switching in IDEs:** When switching between unrelated files or tasks, the semantic manifold shifts. TSP evicts the previous task's isolated tokens while preserving protected "sink" tokens (such as global project instructions and system prompts).
*   **Reducing Context Drag:** Continuously pruning irrelevant snippets prevents the context window from growing indefinitely, keeping inference times and token costs stable during long sessions.

## Performance Profile (Apple Silicon)

| Sequence Length | Avg. Update Latency |
| :--- | :--- |
| 128 tokens | ~40ms |
| 512 tokens | ~615ms |
| 2048 tokens | ~10.9s |

*Note: Latency is proportional to sequence length due to the $O(N^2)$ nature of full spectral bisection. In practice, graph updates are deferred to intervals (e.g., every 64 tokens).*

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

## License
MIT License.
