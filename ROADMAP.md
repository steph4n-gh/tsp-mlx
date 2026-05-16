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

## v3.0 and v4.0: The Dual-System Fractal Architecture

v3.0 and v4.0 work in tandem to completely solve context bloat by splitting the AI's memory into two distinct mechanisms: one for exact factual recall, and one for contextual intuition.

### Implementation Plan:
1.  **Topological Paging (Factual Recall):** When the `tau-gate` Fiedler Vector identifies an isolated "Thought Island" that must be pruned to save VRAM, it compresses that island into a single dense **Macro-Token** that remains in active VRAM. The raw tokens are saved to system RAM/Disk. If the agent later needs exact factual recall (e.g., "Dr. Xylophone Quasar"), attention naturally flows back to the Macro-Token, triggering the framework to "unpack" the raw tokens back into VRAM instantly.
2.  **Test-Time Training (Contextual Intuition):** While the raw facts go to Paging, the mathematical *relationships* and *style* of the evicted tokens are distilled directly into the model's background weights. The engine temporarily halts generation and runs a fast 10-step backward pass exclusively on injected `LoRALinear` value projections (r=64) in the final layer. This ensures the agent retains the "vibe" and implicit context across sessions without taking up a single token of VRAM.
3.  **Persistent Learning:** The updated LoRA adapters can be explicitly serialized to disk (`.safetensors`) by the frontend application. When the API server or script boots, it can optionally load these adapters, ensuring the agent retains its consolidated intuition across isolated sessions as an "Immutable Agent".