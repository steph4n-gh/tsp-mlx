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

### 3. Test-Time Training (TTT) Logic (Resolved)
*   **The Problem:** The initial TTT implementation utilized a simplified proxy task, incorrectly mapping keys directly to values rather than capturing the actual projection from hidden states.
*   **The Resolution:** The consolidation engine was overhauled to use the real hidden states (`x`) for context distillation. The model now performs mathematically rigorous LoRA updates on value projections, successfully 'memorizing' evicted context.

---

## Next Steps for v2.0 (The Foundation)

1.  **Multi-Model Verification:** Prove the universal RoPE patcher works across varied architectures (e.g., Llama-3, Mistral) alongside the existing Qwen2.5 integration.
2.  **Architecture Split (The MLX Graph Barrier):** *Architectural Decision:* We evaluated using `nanobind` to expose the C++ `KVCacheManager` directly to Python. However, we identified a hard Application Binary Interface (ABI) boundary: passing an `mlx::core::array` across the language boundary via memory buffers forces immediate tensor materialization. This breaks the lazy-evaluation computation graph, which is an absolute requirement for computing LoRA gradients during Test-Time Training. 
    *   **Resolution:** The architecture has been decoupled into two distinct, optimized stacks. 
    *   **Python Stack:** Utilizes the hardened Python `KVCacheManager` which orchestrates graph execution natively within MLX Python, while offloading the intensive Fiedler vector graph bisection to the Rust engine (`tau-gate`) via a zero-latency C-FFI (`ctypes`) bridge. This serves as the standard integration path for `mlx-lm`.
    *   **C++ Stack:** The C++ `KVCacheManager` is fully implemented with native gradients and is reserved exclusively for high-performance, pure C++ inference pipelines.

---

## v3.0: Variance Compression (Macro-Tokens)

Currently, TSP deletes isolated context. In v3.0, we compress it. This provides an LLM with the ability to compress a large context block into dense tokens without losing access to the underlying statistical variance.

### Implementation Plan:
1.  **Boundary Detection (The Engine):** Utilize $\tau$-Gate's Fiedler Vector to identify the boundaries of a single, cohesive concept (a "Thought Island").
2.  **Variance Compressor (The Core):** Implement a statistical compressor that computes the variance of hidden states within the island to generate a weighted average, projecting the island into a dense representation.
3.  **Active Cache Synthesis:** When $\lambda_2$ drops, extract the Thought Island from the KV Cache. Pass it through the compressor to generate dense **Macro-Tokens**.

---

## v4.0: Memory Consolidation (Test-Time Training)

v4.0 enables transferring knowledge from short-term memory (KV Cache) to long-term memory (LoRA Weights).

### Implementation Plan:
1.  **Salience Trigger:** Modify $\tau$-Gate to score isolated islands based on "Salience" (the density of internal connections). 
2.  **The Consolidation Hook:** If a highly salient island shifts out of the active context, we trigger an "Online Consolidation" event.
3.  **Dynamic Auto-Tuning:** The engine detects the quantization level of the base model to dynamically configure gradient clipping and learning rates.
4.  **Localized Backward Pass:** The MLX engine temporarily halts generation and runs a few steps of a backward pass exclusively on injected `LoRALinear` value projections, using the Thought Island as the training data.
5.  **Persistent Learning:** The updated LoRA adapters are serialized to disk (`tsp_adapters.safetensors`). When the API server restarts, it automatically loads these adapters, ensuring the agent retains its consolidated knowledge across sessions.