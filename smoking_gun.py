import mlx.core as mx
from mlx_lm import load
import time
import platform

print("\n\033[1;33m--- THE SMOKING GUN --- \033[0m")
print(f"Architecture: {platform.machine()}")
print(f"Metal GPU:    {mx.metal.is_available()}")
print("-----------------------\n")

print("Loading model directly into VRAM...")
model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
mx.eval(model.parameters())

dummy_ids = mx.zeros((1, 8192), dtype=mx.int32)

print("\u23F3 [Phase 1] Forcing Apple Silicon JIT Compile (Off the clock)...")
@mx.compile
def forward(ids):
    logits = model(ids)
    return logits[:, -1, :]

_ = forward(dummy_ids)
mx.eval(_) 

print("\u26A1 [Phase 2] Blasting pure Metal hardware...")
start = time.time()
_ = forward(dummy_ids)
mx.eval(_)
elapsed = time.time() - start

print(f"\n\033[1;32mTRUE HARDWARE SPEED: {8192 / elapsed:.0f} tok/s\033[0m\n")