import mlx.core as mx
from mlx_lm import load
import time
import platform

print("\n--- HARDWARE DIAGNOSTIC ---")
print(f"System Architecture: {platform.machine()}")
print(f"Metal GPU Enabled:   {mx.metal.is_available()}")
print("---------------------------\n")

print("Loading model directly into VRAM...")
model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
mx.eval(model.parameters())

print("Generating 3,000 dummy tokens...")
dummy_ids = mx.zeros((1, 3000), dtype=mx.int32)

print("Executing pure, raw Metal forward pass...")
start = time.time()
_ = model(dummy_ids)
mx.eval(_)
elapsed = time.time() - start

print(f"\n\u26A1 Raw Hardware Speed: {3000 / elapsed:.0f} tok/s\n")
