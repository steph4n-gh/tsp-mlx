# Gemini CLI: Developer Guide for τ-Spectral Pruner (tsp-mlx)

This repository contains the high-performance C++ MLX router (`tsp-mlx`) for the τ-Gate Neuro-Symbolic Paging system.

## 🏛️ Core Architectural Mandates
1.  **C++ Native Core:** This is a C++ project. The core logic relies on the MLX C++ API and a static library integration of the Rust `tau-gate` engine via C FFI.
2.  **Zero-Latency FFI:** Do not use subprocesses or JSON over `stdin`/`stdout`. Communication with the mathematical bisection engine must happen via the `tau_gate_analyze` C-interface.
3.  **VRAM Efficiency over Everything:** Graph edge tracking is handled in C++ system memory (`std::set`). Never instantiate dense NxN attention matrices in memory during generation.
4.  **Universal RoPE Patching:** Models implement Rotary Position Embeddings (RoPE) differently. Modifications must dynamically support fragmented, non-contiguous absolute position IDs.

## 🌿 Git Workflow Rules
1.  **Never Push to Main:** You must NEVER push code directly to the `main` branch. 
2.  **Always Use PRs:** All code changes must be pushed to a feature branch, and a Pull Request must be opened using the `gh` CLI.
3.  **Documentation Exception:** The ONLY exception is documentation updates (e.g., `README.md`, `docs/*.md`). Documentation changes CAN be pushed directly to `main`.

## ⚖️ Licensing
This repository is licensed under the **MIT License**. It is intended to be open and permissive for the MLX community. Do not introduce proprietary or restrictive licenses here.
