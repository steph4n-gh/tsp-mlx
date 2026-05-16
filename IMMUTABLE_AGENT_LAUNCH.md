# The Space Complexity Trap: How Math Solves LLM Amnesia and Supply Chain Attacks

**Two Groundbreaking Problems, Solved.**

💥 **TSP:** Runs autonomous LLM agents infinitely on a Mac without Out-of-Memory (OOM) crashes or RAG databases, using **Topological Compression** and **Continuous Memory Consolidation**.

💥 **τ-Gate:** Kills **Dependency Confusion and Typosquatting** attacks before they install, using topological math instead of reactive CVE databases.

They look like two completely different problems (Generative AI memory vs. Cybersecurity). Structurally, they are the exact same problem: **failures in managing massive, chaotic graphs.**

We built two source-available frameworks to solve them using the exact same mathematical truth: **Spectral Graph Theory.**

Here is how they work, and why they are the foundation for the **"Immutable Agent."**

---

### Problem 1: Transitive Secrecy & τ-Gate 🛡️

Vulnerability scanners rely on CVE databases. They are inherently reactive. If a hacker creates a new, unknown malicious package and hides it via **Dependency Confusion** deep in a package tree, traditional scanners won’t see it.

We built **τ-Gate** to stop Zero-Day supply chain attacks before they execute.

**The How:**
Instead of scanning for known bad code, τ-Gate analyzes the **Shape of Trust**. It builds an in-memory graph of your package tree directly from the network registry. Trusted software forms a dense mainland of connections. Attackers inevitably form tiny, isolated islands (Topological Islands) trying to hide.

τ-Gate calculates the Graph Laplacian and finds the **"Spectral Gap"** (the sharpest drop-off in connectivity). If an isolated island attempts to execute a build script (like a `postinstall` hook), τ-Gate **mathematically flags** it as anomalous and severs the connection, blocking the installation before a single line of malicious code reaches your disk.

**The Result:** Mathematical, Zero-Trust defense against structural supply chain attacks, with zero external dependencies (pure Rust).

---

### Problem 2: Context Bloat & TSP 🚀

Local LLMs are built for chatbots, not agents. They load every token into VRAM. When VRAM fills up, they crash. The industry "fix" is RAG (Vector Databases)—but RAG is slow, lossy, and destroys the structural integrity of code.

We built **TSP (τ-Spectral Pruner)** for MLX. It is an organic memory engine designed specifically for the Unified Memory architecture of Apple Silicon.

**The How:**
TSP intercepts the LLM's attention matrix and treats it as a graph. When VRAM gets tight, TSP uses the exact same Spectral Bisection math as τ-Gate to find "Thought Islands"—chunks of context (like an old log file) that the AI is no longer thinking about.

Instead of throwing them away, TSP implements a **Dual-System Fractal Memory**:
1. **Topological Paging (The Conscious Memory):** The isolated context is paged to system RAM, leaving only a dense **Macro-Token** anchor in active VRAM. If the agent later pays high attention to that Macro-Token, the framework instantly "unpacks" the raw tokens back into VRAM for flawless factual recall.
2. **True Test-Time Training (The Subconscious Memory):** Simultaneously, the engine runs 10 fast steps of gradient descent via LoRA (TTT) on the evicted tokens. This physically burns the "vibe", logic, and style of the forgotten context into the model's background neural weights, granting permanent intuition without consuming a single token of context budget.

**Because the mathematical bisection is offloaded to a zero-dependency Rust daemon via a direct C-FFI bridge, the $O(E)$ sparse iterative solver executes in microseconds. The LLM maintains native Apple Silicon generation speeds while memory is dynamically swapped and trained.**

**The Result:** A 7B parameter agent can run endlessly on a MacBook Pro, read thousands of files, and never OOM. It achieves cluster-level context scaling locally.

---

### The Synergy: The Immutable Agent ⚡

While τ-Gate secures traditional software and TSP empowers generative AI, combining them creates something entirely new: **The Immutable Agent Operating System.**

When you give an autonomous agent the keys to your terminal, you are giving a stochastic machine lethal power. We provide the mathematical cage.

**Internal Security (Semantic Firewall):**
TSP monitors the agent's attention graph in real-time. If a foreign context (like a hijacked webpage the agent is reading) forms a topological island that suddenly exhibits anomalous, aggressive edge density pointing directly at the model's **System Prompt** (the core rules), TSP **topologically isolates** it as a Prompt Injection attempt and drops a **FATAL_BLOCK**, halting generation before the weaponized command is even typed.

**External Security (Hypervisor Gate):**
When the agent decides to use a tool to execute a terminal command (e.g., `npm install`), the Python wrapper intercepts the request. It forces the agent to ask the **τ-Gate Hypervisor** for permission. τ-Gate audits the requested package's topology; if it detects a structural anomaly, it feeds the block back to the LLM. The agent, now aware of the security risk, can autonomously pivot its strategy to a secure alternative.

**TSP** gives the agent the "Infinite Brain" to autonomously code forever on edge hardware.
**τ-Gate** guarantees the agent will never poison its own environment.

This isn’t a wrapper. It’s an operating system for persistent, secure, local AI.

*The math is solid, the code is open, and the demos are ready to run.*

⭐ **Read the math, star the repos, and run the Immutable Agent on your Mac today:**
🧠 **TSP (Neuro-Symbolic Memory):** [https://github.com/steph4n-gh/tsp-mlx](https://github.com/steph4n-gh/tsp-mlx)
🛡️ **τ-Gate (Topological Firewall):** [https://github.com/steph4n-gh/tau-gate](https://github.com/steph4n-gh/tau-gate)
