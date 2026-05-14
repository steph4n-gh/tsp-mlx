# TSP (τ-Spectral Pruner) Roadmap & Architectural Reality

## The Goal: Topologically Persistent Context
The goal of the `tsp-mlx` framework is to enable "Infinite Context" (Persistent memory) for LLMs by dynamically pruning useless tokens from the KV Cache based on mathematical graph theory (Spectral Bisection of the Attention Matrix).

## Architectural Realities & Resolutions

### 1. The Math Latency & FFI Bridge
*   **The Problem:** Calculating the Fiedler vector on an 8,000 x 8,000 matrix via an NDJSON OS pipe created unacceptable latency.
*   **The Resolution:** We successfully migrated from a UNIX subprocess to a **Native C FFI Bridge**. The core dependency, `tau-gate`, is now compiled as a static library (`libtau_gate.a`) and linked directly into the C++ engine. Latency is now $O(1)$ memory access.

### 2. MLX and the RoPE Hack (Solved)
*   **The Problem:** High-level wrappers assume a contiguous text stream (`offset + arange`). Deleting tokens mid-generation causes severe spatial hallucinations.
*   **The Resolution:** The C++ Core (and the Python wrapper) now implement a Universal RoPE Patcher that intercepts the attention extraction *before* the dot product, applying rotation using the explicit, fragmented `position_ids` array from the `SparsePositionTracker`.

### 3. Space Complexity & Graph Memory
*   **The Problem:** Storing massive edge lists in Python `List` objects caused O(N^2) memory leaks.
*   **The Resolution:** Migrated graph tracking to C++ `std::set` (and Python `set` comprehensions), completely eliminating duplicate edge accumulation and GPU VRAM spikes.

---

## Next Steps for v2.0

1.  **Multi-Model Verification:** Prove the universal RoPE patcher works across varied architectures (e.g., Llama-3, Mistral) alongside the existing Qwen2.5 integration.
2.  **Native C++ Bindings (PyBind11):** Currently, the Python LLM implementation (`demo_chat.py`) uses a backported Python tracking loop. The next step is to use `pybind11` to expose the hardened `tsp::KVCacheManager` C++ class directly to Python, allowing `mlx-lm` users to achieve maximum TPS.
3.  **Prompt Injection / Anomaly Thresholding:** Refine the "Executive Function" logic to allow host applications to intercept generation when the $\lambda_2$ (Algebraic Connectivity) drops below a configurable security threshold.