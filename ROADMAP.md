# TSP (τ-Spectral Pruner) Roadmap & Architectural Reality

## The Goal: Infinite Context
The goal of the `tau-mlx-router` is to enable "Infinite Context" for LLMs by dynamically pruning useless tokens from the KV Cache based on mathematical graph theory (Spectral Bisection of the Attention Matrix).

## Architectural Sanity Check (Feedback)

### 1. Market Reality vs. Security Pitch
*   **The Pitch:** Real-time memory poisoning defense and semantic threat interception.
*   **The Reality:** The primary, immediate value proposition is **The Paging Router**. It saves VRAM by deleting useless context, allowing models to run infinitely on constrained Apple Silicon.

### 2. MLX and the RoPE Hack (The Hardest Technical Hurdle)
*   High-level wrappers like `mlx-lm.generate` assume a standard, contiguous text stream and handle Rotary Position Embeddings (RoPE) automatically.
*   **The Reality:** To delete tokens mid-generation and pass in a custom, sparse array of `position_ids`, we **must** fork or rewrite the inner decoding loop. We cannot use out-of-the-box MLX generation functions. We must manually pass the modified cache and position indices to the transformer block.

### 3. The Math Latency
*   Calculating the Fiedler vector on an 8,000 x 8,000 matrix is fast, but not free.
*   **The Reality:** We cannot run the Rust daemon on every single token, or we will destroy Tokens-Per-Second (TPS). We must batch the evaluation (e.g., `eval_interval = 64`), which is currently implemented in `cortex_hook.py`.

### 4. The UNIX Daemon (Validation)
*   **The Reality:** Embedding Rust into Python via PyO3 introduces massive complexity (C-ABI debugging, memory leaks, GPU state clashes).
*   Using NDJSON over `stdin`/`stdout` is bulletproof. The core TSP engine remains perfectly isolated.

---

## Next Steps for v4.0

1.  **Select a Model:** Download a standard open-weights model via `mlx-lm` (e.g., Llama-3-8B-Instruct).
2.  **Rewrite the Inference Loop:** Extract the low-level `generate_step` function from MLX.
3.  **Inject the Router:** Modify the loop to pass the raw attention matrices to `KVCacheManager.update()`.
4.  **Handle RoPE:** Ensure the transformer's RoPE application accepts the fragmented `position_ids` array from `SparsePositionTracker` instead of assuming a contiguous `[0, 1, 2, ...]` array.