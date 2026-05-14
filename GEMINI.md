# Gemini CLI: Developer Guide for τ-Spectral Pruner (tsp-mlx)

This repository contains the Python-based MLX router (`tsp-mlx`) for the τ-Gate Neuro-Symbolic Paging system.

## 🏛️ Core Architectural Mandates
1.  **Apple Silicon & MLX Only:** This package is strictly designed for the `mlx` and `mlx-lm` frameworks. Do not introduce PyTorch, TensorFlow, or JAX dependencies.
2.  **No Rust Code Here:** This repository is purely Python. The core mathematical bisection logic lives in the `tau-gate` repository. We integrate with it strictly via the `CortexHook` subprocess using NDJSON over `stdin`/`stdout`.
3.  **VRAM Efficiency over Everything:** When intercepting attention matrices, always extract sparse, 1D rows token-by-token (as implemented in `cortex_hook.py`) to avoid O(N^2) VRAM spikes on the GPU. Never instantiate dense NxN attention matrices in memory during generation.
4.  **Universal RoPE Patching:** Models implement Rotary Position Embeddings (RoPE) differently (e.g., NeoX vs. GPT-J). Any modifications to `rope_patches.py` must dynamically support both `traditional=False` and `traditional=True` interleaving without hardcoding model names.

## ⚖️ Licensing
This repository is licensed under the **MIT License**. It is intended to be open and permissive for the MLX community. Do not introduce proprietary or restrictive licenses here.
