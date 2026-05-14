# What is TSP? (The Definitive Summary)

This document serves as the official positioning statement for the $\tau$-Spectral Pruner (TSP). It explains exactly what the product is, what it does, and who it is for, scaling from absolute simplicity to expert-level technical detail.

---

## 1. The One-Sentence Pitch (The Hook)
TSP is a C++ memory manager that allows Large Language Models to run infinitely on constrained hardware by automatically throwing away irrelevant context while preserving core instructions.

---

## 2. The Elevator Pitch (The Paragraph)
Standard LLMs have a fatal flaw: their working memory (the KV Cache) grows linearly with every word they read or write, eventually causing the system to run out of VRAM and crash. The $\tau$-Spectral Pruner (TSP) solves this Space Complexity Trap by treating the model's memory as a mathematical graph rather than a simple queue. Using spectral graph theory, TSP runs entirely in the background, constantly identifying and deleting isolated "islands" of irrelevant noise from the GPU, ensuring the active context window remains artificially small and hyper-focused. This provides developers with "Topologically Persistent Context"—the ability to run autonomous AI agents continuously for days or weeks without memory collapse or instruction amnesia.

---

## 3. Explain Like I'm 6 (The Intuition)
Imagine an AI is a student taking a really, really long test. To answer the questions, the student has to keep all the information they've read inside a tiny backpack (their memory). 

Normally, when the backpack gets full, the student either has to stop taking the test, or they have to throw away the *oldest* notes—even if those notes contain the most important instructions!

**TSP is a smart organizer for the backpack.** Instead of throwing away the oldest notes, it looks at everything in the bag and figures out which notes are completely unrelated to what the student is thinking about *right now*. It throws away the useless distraction notes (like a random math formula during a history essay) so the student never runs out of room and never forgets the important instructions.

---

## 4. For the Experts (The Technical Reality)
TSP is a low-level, natively compiled C++ extension for the MLX framework, backed by a zero-dependency Rust mathematical solver (`tau-gate`) linked via a zero-latency C FFI bridge. 

**What it does:** 
It intercepts the $O(N^2)$ attention mechanism of the LLM's final transformer layer *after* causal masking but *before* sequence decoding. It builds a directed graph of attention probabilities in C++ system memory (`std::set`). On a scheduled interval, the Rust engine performs an eigenvalue decomposition on the graph's Laplacian, isolating the Fiedler vector ($\lambda_2$) to identify the Maximum Spectral Gap. 

**The result:**
Nodes (tokens) that fall into a disconnected semantic manifold are surgically evicted from the physical MLX KV Cache tensors in-place. A custom C++ Universal RoPE Patcher intercepts subsequent generation steps, applying Rotary Position Embeddings based on the fragmented absolute position IDs (rather than contiguous physical indices), completely eliminating spatial tearing and preventing the hallucination death-spiral associated with mid-sequence cache truncation.

---

## 5. Who is this for? (The Target Audience)

TSP is built for engineers and enterprises pushing the limits of local, persistent AI. 

*   **Builders of Autonomous Agents (Devin, AutoGPT clones):** If you are building an agent that loops for hours reading logs, writing code, and traversing directories, TSP is the only way to prevent the agent from eventually OOM-crashing or "forgetting" its system prompt due to sliding-window context ejection.
*   **Edge & Mobile AI Developers:** If you are deploying LLMs to MacBooks, iPhones, or edge devices where Unified Memory is strictly capped, TSP acts as a hard VRAM ceiling. You can compress a 50K token conversation into a persistent 4K footprint.
*   **IDE and CLI Integrations:** For tools like Cursor, Claude Code, or Gemini CLI, TSP acts as a background noise filter. It aggressively deletes terminal spam and old file scratchpads from the active context, radically reducing Time-To-First-Token (TTFT) and API/Compute costs over long, multi-file developer sessions.
*   **Security Teams (CISOs):** TSP's spectral engine acts as an "Executive Function." By monitoring the algebraic connectivity ($\lambda_2$) of the graph, it provides a mathematical foundation for detecting abrupt semantic shifts associated with prompt injection attacks.
