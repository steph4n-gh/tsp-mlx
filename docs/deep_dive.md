# TSP Deep Dive: Configuration, Context, and Multi-Turn Survival

This is a strictly technical deep dive into how the $\tau$-Spectral Pruner (TSP) operates, how to configure it, and why it doesn't catastrophically break LLMs during generation.

## 1. Configuring TSP with an LLM

TSP is not a wrapper around `mlx-lm.generate`. It is a low-level injection into the inference loop. To use it, you must bypass the high-level generation functions and manage the KV cache manually.

### The Setup Phase
1.  **Initialize the Manager:** You create a `KVCacheManager` connected to the `CortexHook` (the FFI bridge to the Rust math engine).
2.  **Patch RoPE:** You must call `patch_rope_for_sparse_positions(model, tracker)`. Standard MLX models assume token 50 is physically at index 50. TSP breaks this assumption. The patch forces the model to look up the *true semantic position* of a token from the `SparsePositionTracker` before applying Rotary Position Embeddings.
3.  **Patch Attention:** You call `patch_attention_for_extraction(model)`. This intercepts the `q_proj` and `k_proj` outputs in the final transformer layer, applies RoPE to them, calculates the Causal Mask and Softmax, and saves the normalized probabilities so the Rust engine can read them.

### The Inference Loop
During generation, instead of a standard step, you explicitly call the TSP update:
```python
# 'attn_matrix' is extracted from the patched attention layer
# 'sinks' protects the first N tokens (your system prompt)
pruned_caches = kv_manager.update(attn_matrix, raw_caches, sinks=[0, 1, 2, 3, 4])
```
If the spectral engine finds an isolated island, it mutates the MLX cache tensors in-place, truncating the sequence length, and updates the `SparsePositionTracker`.

## 2. What is Kept vs. Discarded (And Why)

TSP does not care about the *age* of a token. It cares exclusively about **Algebraic Connectivity**.

### The Mechanism
1.  **The Graph:** Every token is a node. If Token A attends heavily to Token B, an edge is drawn. 
2.  **The Fiedler Vector:** The Rust engine calculates the second smallest eigenvalue ($\lambda_2$) of the Graph Laplacian. This tells us how "connected" the whole conversation is.
3.  **The Bisection:** If the conversation fractures into two distinct topics (e.g., you stop talking about Python and start talking about pizza), the graph splits. The eigenvector values will naturally sort the tokens into two clusters separated by a "Spectral Gap."

### What gets discarded?
The cluster of tokens that the model is *currently ignoring*. If the current token's attention is focused heavily on the pizza discussion, the Python tokens become a mathematically isolated "Island." TSP evicts the island.

### What is kept?
*   **The Current Manifold:** The tokens relevant to the immediate thought process.
*   **Sinks:** The `sinks` array (configured by the developer) explicitly protects indices from eviction. You **must** pass the indices of your System Prompt and ChatML formatting headers into the `sinks` array, or the model will forget its identity.

## 3. The Multi-Turn Chat Paradox

**The Developer's Fear:** 
*"If I have a 20-turn conversation using Llama-3, the context is full of strict structural tags like `<|start_header_id|>user<|end_header_id|>`. If TSP aggressively deletes tokens out of the middle of the context, won't it delete half of a formatting tag, leaving `<|end_header_id|>` floating alone? Won't that instantly break the LLM?"*

**The Reality:** 
No. It does not break, and multi-turn chats survive. Here is the mathematical reality of why:

### A. Attention dictates Survival
If a specific structural tag from Turn 4 is critical for the LLM to understand how to format its output on Turn 20, the LLM will *attend* to it. If the LLM attends to it, a strong edge is drawn in the graph. Therefore, the tag is mathematically connected to the current generation and **will not be pruned**. 
If the tag *is* pruned, it is because the causal mask and softmax probabilities proved the LLM was completely ignoring it anyway. The LLM does not need what it does not look at.

### B. The RoPE Patcher Fixes Spatial Tearing
If TSP deletes 500 tokens from the middle of a chat log, the tokens on either side of the gap are now physically adjacent in the KV cache tensor. In a standard LLM, this causes "Spatial Tearing"—the model thinks the user's question from Turn 2 was asked immediately after Turn 10.
Our `patch_rope_for_sparse_positions` fixes this. It ensures that the token from Turn 2 still rotates mathematically as if it is at Position 100, and the token from Turn 10 still rotates as if it is at Position 600. The "distance" between them is preserved perfectly, even though the physical memory between them was deleted.

### C. The Local Horizon
LLMs rely heavily on the *most recent* formatting tags to understand their current role. The tokens comprising the current user prompt and the current assistant prefix are inside the active "Thought Island" and are inherently protected by density of attention. 

**Conclusion:** The LLM's own attention mechanism acts as the ultimate filter. If a piece of context—whether a word, a code snippet, or a structural tag—is necessary for the current generation, the LLM's attention graph will protect it from the Spectral Pruner.