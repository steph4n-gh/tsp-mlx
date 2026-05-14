# Theoretical Foundation: Nested Learning & TSP

The $\tau$-Spectral Pruner (TSP) architecture strongly aligns with the theoretical framework presented in the paper *Nested Learning: The Illusion of Deep Learning Architecture* (Google Research).

This document outlines how TSP implements the core principles of Nested Learning to achieve Topologically Persistent Context and overcome the limitations of standard LLM memory management.

## 1. Memory as an Active, Self-Referential Graph
**Nested Learning Principle:** The paper argues against the "Passive Landfill" view of memory, positing that learning and memory are active "Associative Memories" compressing their own context. It introduces architectures that generate their own keys and values to actively adapt to context.

**TSP Implementation (The Executive Function):** 
TSP implements this exact self-referential adaptivity. Standard MLX generation treats the KV Cache (the attention mechanism) as a passive queue. TSP re-frames attention as a **Topological Graph**. By passing the attention matrix to the $\tau$-Gate daemon and calculating the Fiedler vector ($\lambda_2$), TSP actively curates its own working memory. It modifies its own state (the KV Cache) in real-time based on the mathematical shape of its own thoughts, pruning isolated "Thought Islands."

## 2. The Multi-Timescale Paradigm
**Nested Learning Principle:** The brain does not have a rigid "short term" vs. "long term" memory block. Instead, it utilizes a spectrum of update frequencies (Gamma, Beta, Theta waves). Fast layers adapt quickly but forget quickly; slow layers update rarely but hold persistent knowledge.

**TSP Implementation:** 
TSP manages the KV Cache using a multi-timescale approach:
*   **High-Frequency (The Token Stream):** The MLX engine generates and processes tokens rapidly.
*   **Adaptive Low-Frequency (The Spectral Bisection):** $\tau$-Gate evaluates global connectivity on a dynamic interval (e.g., every 16 to 128 tokens). This frequency adapts based on the volatility of the Algebraic Connectivity score ($\lambda_2$), checking less often when the semantic manifold is stable, and more often during abrupt topic shifts.
*   **Persistent Knowledge (Sinks):** Protected system instructions ("Sinks") are explicitly prevented from eviction, acting as the zero-frequency, permanent identity of the agent.

## 3. Solving the Context Trap via In-Context Compression
**Nested Learning Principle:** Current LLMs suffer from "Anterograde Amnesia" because they cannot transfer fast, in-context learning into slow, persistent weights without catastrophic forgetting.

**TSP Implementation:** 
Rather than designing a new neural network architecture, TSP solves this problem practically for existing open-weights models. TSP uses Spectral Graph Theory to *compress the in-context learning*. By throwing away the mathematically irrelevant noise, TSP ensures the active context window never overflows. This allows standard models (like Llama 3 or Qwen) to effectively achieve infinite conversation lengths without altering their pre-trained MLP weights.

## 4. Fuzzy Sinks & Dynamic Persistence
**Nested Learning Principle:** Memory persistence is distributed and interconnected, rather than statically assigned to specific blocks.

**TSP Implementation:** 
TSP implements "Fuzzy Sinks." While root System Prompts are statically protected, any active token that receives overwhelming, sustained attention from the current generation—and is highly connected back to the Root Sinks—dynamically inherits protection. This allows the core identity and rules of the agent to evolve fluidly during a session without being statically hardcoded.
