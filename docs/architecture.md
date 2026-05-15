# TSP Architecture

The τ-Spectral Pruner (TSP) manages the KV Cache by modeling the attention matrix as a directed graph and applying eigenvalue decomposition to identify isolated semantic clusters.

## 1. Python and Rust Integration

TSP uses Python and MLX for tensor operations and Rust for the mathematical solver.

*   **Memory Management:** TSP uses the MLX Python API to modify the KV cache tensors. Graph edges are tracked using Python sets to prevent memory leaks from duplicate edge accumulation.
*   **FFI Bridge:** The `tau-gate` spectral solver is compiled as a Rust dynamic library. It exposes a C interface (`tau_gate_analyze`) that is called directly by the Python engine via `ctypes`. This avoids the serialization and latency of passing JSON over standard I/O.

## 2. Topologically Persistent Context

TSP relies on a topological heuristic to decide which tokens to evict. 

*   **Spectral Bisection:** It calculates the Fiedler vector (the eigenvector corresponding to the second smallest eigenvalue, $\lambda_2$, of the graph's Laplacian). This vector is used to find the maximum spectral gap, bisecting the graph into connected components.
*   **Eviction:** Tokens belonging to disconnected components (islands) are evicted from the MLX KV cache.
*   **RoPE Alignment:** Rotary Position Embeddings (RoPE) require contiguous positional indices. TSP patches the RoPE implementation to apply rotation using the tracked, non-contiguous absolute position IDs before dot product calculation. This maintains spatial accuracy in the attention matrix after intermediate tokens are removed.

## 3. Executive Function and Safety

*   **Sink Protection:** The Python manager implements a hard-stop list of "Sink Tokens" (usually the initial system instructions). Regardless of the daemon's output, sinks are never pruned from the KV cache.
*   **Anomaly Detection:** Tracking the algebraic connectivity ($\lambda_2$) provides a metric for the graph's overall cohesiveness. Sudden drops in connectivity can indicate abrupt topic changes or potential prompt injection, allowing the host application to block execution.
