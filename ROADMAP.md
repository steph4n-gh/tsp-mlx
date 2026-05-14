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

## Next Steps for v2.0 (The Foundation)

1.  **Multi-Model Verification:** Prove the universal RoPE patcher works across varied architectures (e.g., Llama-3, Mistral) alongside the existing Qwen2.5 integration.
2.  **Native C++ Bindings (PyBind11) [CRITICAL BLOCKER]:** *The "No BS" Reality Check:* Currently, the Python LLM implementations (`demo_chat.py` and `scripted_chat.py`) use a backported Python tracking loop. The advanced V3 (Compression) and V4 (Consolidation) features are scaffolded natively in the `cpp/` directory to guarantee performance, meaning they are fundamentally disconnected from the Python `mlx-lm` ecosystem right now. The absolute highest priority is using `pybind11` to expose the hardened `tsp::KVCacheManager` C++ class directly to Python. Until this is done, V3 and V4 cannot be used in live chat interfaces.

---

## v3.0: Topological Compression (Macro-Tokens)

Currently, TSP deletes isolated context. In v3.0, we will compress it. This provides an LLM with "Fractal Memory"—the ability to compress a 100,000-line codebase into 100 dense tokens without losing access to the underlying logic.

### Implementation Plan:
1.  **Boundary Detection (The Engine):** Utilize $\tau$-Gate's Fiedler Vector to identify the exact, mathematically perfect boundaries of a single, cohesive concept (a "Thought Island"). This solves the fatal flaw of arbitrary "chunking" algorithms.
2.  **Sub-Manifold Autoencoder (The Core):** Implement a localized, ultra-fast neural network (a mini-transformer or deep MLP) using MLX to project sequences of KV cache tensors into a single token vector.
3.  **Active Cache Synthesis:** When $\lambda_2$ drops (signaling a semantic shift), extract the 400-token Thought Island from the KV Cache. Pass it through the autoencoder to generate 1 dense **Macro-Token**.
4.  **In-Place Matrix Stitching:** Replace the 400 raw tokens in the KV Cache tensors with the 1 Macro-Token.
5.  **RoPE Realignment:** Update `SparsePositionTracker` so that the single Macro-Token inherits the spatial weight of the original 400 tokens, ensuring subsequent tokens rotate perfectly and spatial continuity is preserved.

---

## v4.0: Memory Consolidation (Test-Time Training)

Inspired by *Nested Learning* (Section 1.1: Human Brain Perspective), v4.0 will solve "Anterograde Amnesia" by transferring knowledge from short-term memory (KV Cache) to long-term memory (MLP Weights).

### Implementation Plan:
1.  **Salience Trigger:** Modify $\tau$-Gate to not only detect isolated islands, but to score them based on "Salience" (the density of internal connections). 
2.  **The Consolidation Hook:** If a highly salient island (e.g., the user teaching the AI a complex new API) shifts out of the active context, we do not just compress it. We trigger an "Online Consolidation" event.
3.  **The M3 Optimizer (Multi-scale Momentum Muon):** Standard optimizers (like AdamW) cause catastrophic forgetting in continual learning. Following the *Nested Learning* paper (Section 7.2), we must implement the M3 Optimizer natively in MLX C++. This optimizer maintains long-context momentum, ensuring that baking new knowledge does not destroy old knowledge.
4.  **Localized Backward Pass:** The MLX engine temporarily halts generation and runs a low-rank backward pass (using the M3 Optimizer) exclusively on the LLM's MLP blocks, using the Thought Island as the training data.
5.  **Persistent Learning:** The knowledge is permanently baked into the model's weights. The KV Cache is instantly cleared, and the agent has permanently learned the new skill without requiring a massive, offline fine-tuning run.