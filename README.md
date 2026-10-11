# τ-Spectral Pruner (TSP) for MLX

TSP is an experimental MLX memory-management and attention prototype. This repository contains a Python package, a separate C++ prototype, demos, and design documents. Claims about unlimited context, lossless recall, inference overhead, and prompt-injection protection in the older documents have not been established by the tests in this checkout.

## Current checkout status

The Python package currently has an import blocker: `tsp_mlx/sparse_cache.py` annotates `model: nn.Module` but does not import `nn`. Because `tsp_mlx/__init__.py` imports that module, the generation example below should be treated as intended API, not a verified quick start. The repository's only checked-in Python test file uses synthetic attention arrays; it is not a performance or model-quality evaluation.

The C++ build is separate from the Python package. Its [CMake configuration](cpp/CMakeLists.txt) refers to a `../../supplychain/target/release/libtau_gate.a` file and a specific `venv/lib/python3.14` location. Those paths are not supplied by this checkout.

## Python setup

On a compatible Apple Silicon environment, the dependencies declared in [pyproject.toml](pyproject.toml) can be installed from the repository root:

```sh
git clone https://github.com/steph4n-gh/tsp-mlx.git
cd tsp-mlx
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

There is no `requirements.txt` in this repository. The import blocker above must be resolved before the package API can be used.

## Intended generation API

The implementation of `generate_with_tsp` is in [tsp_mlx/generate.py](tsp_mlx/generate.py). After the package import issue is fixed, a caller can load an MLX model and pass it to this asynchronous generator:

```python
import asyncio
from mlx_lm import load
from tsp_mlx.generate import generate_with_tsp

async def main():
    model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-8bit")
    async for text, stats in generate_with_tsp(
        model, tokenizer,
        prompt="Write a haiku about a cybernetic dragon.",
        max_tokens=50,
    ):
        print(text, end="", flush=True)

asyncio.run(main())
```

The model download and this example have not been run as part of this documentation update.

## Documentation

- [Architecture](docs/architecture.md)
- [Agent integration guide](docs/agent_integration_guide.md)
- [Research whitepaper](docs/whitepaper.md)
- [Roadmap](ROADMAP.md)

These documents describe design goals and experiments; check implementation and current evidence before relying on operational claims.

## License

[MIT License](LICENSE).
