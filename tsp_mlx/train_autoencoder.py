import os
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from tsp_mlx.compression import SubManifoldAutoencoder
import json

def generate_training_data(model, tokenizer, num_samples=5):
    """
    Generates (island_kv, summary_kv) pairs for Distillation training.
    In a real scenario, this would use a large corpus. For this demo, we use synthetic data.
    """
    print(f"[Trainer] Generating {num_samples} distillation samples...")
    
    samples = [
        "The quick brown fox jumps over the lazy dog. This is a classic pangram used to test typewriters and keyboards.",
        "Quantum entanglement is a physical phenomenon that occurs when a group of particles are generated, interact, or share spatial proximity in a way such that the quantum state of each particle of the group cannot be described independently of the state of the others.",
        "To make a Margherita pizza, you need dough, San Marzano tomatoes, mozzarella cheese, fresh basil, salt, and extra-virgin olive oil.",
        "Isaac Newton's laws of motion are three basic laws of classical mechanics that describe the relationship between the motion of an object and the forces acting on it.",
        "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation."
    ]
    
    training_pairs = []
    
    for text in samples[:num_samples]:
        # 1. Extract KV of the Raw Island
        raw_prompt = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        raw_ids = mx.array(tokenizer.encode(raw_prompt))[None]
        
        raw_caches = make_prompt_cache(model)
        _ = model(raw_ids, cache=raw_caches)
        
        # We just grab the first layer's keys/values for the autoencoder
        k_island = raw_caches[0].keys
        v_island = raw_caches[0].values
        
        # 2. Generate a "Summary" token KV cache
        summary_prompt = tokenizer.apply_chat_template([
            {"role": "user", "content": f"Summarize this in one short sentence: {text}"}
        ], tokenize=False, add_generation_prompt=True)
        
        summary_ids = mx.array(tokenizer.encode(summary_prompt))[None]
        summary_caches = make_prompt_cache(model)
        
        # Forward pass for the prompt
        logits = model(summary_ids, cache=summary_caches)
        
        # Generate exactly ONE summary token
        next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        _ = model(next_token, cache=summary_caches)
        
        # Extract the KV cache of JUST the summary token
        # The cache has shape [batch, heads, seq_len, head_dim]
        # We want the last element of the sequence
        k_summary = summary_caches[0].keys[:, :, -1:, :]
        v_summary = summary_caches[0].values[:, :, -1:, :]
        
        training_pairs.append((k_island, v_island, k_summary, v_summary))
        
    return training_pairs

def train():
    print("[Trainer] Loading model (Llama-3.2-1B-Instruct)...")
    model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    
    # Check head_dim
    # Normally we'd inspect model.args, but for safety we'll generate a dummy cache to inspect
    dummy_cache = make_prompt_cache(model)
    _ = model(mx.array([[0]]), cache=dummy_cache)
    head_dim = dummy_cache[0].keys.shape[-1]
    
    print(f"[Trainer] Detected KV Head Dim: {head_dim}")
    
    autoencoder_k = SubManifoldAutoencoder(hidden_dim=head_dim)
    autoencoder_v = SubManifoldAutoencoder(hidden_dim=head_dim)
    
    mx.eval(autoencoder_k.parameters())
    mx.eval(autoencoder_v.parameters())
    
    optimizer_k = optim.AdamW(learning_rate=1e-3)
    optimizer_v = optim.AdamW(learning_rate=1e-3)
    
    dataset = generate_training_data(model, tokenizer)
    
    def loss_fn_k(model_k, x, y):
        pred = model_k(x)
        return mx.mean(mx.square(pred - y))
        
    def loss_fn_v(model_v, x, y):
        pred = model_v(x)
        return mx.mean(mx.square(pred - y))
        
    loss_and_grad_fn_k = nn.value_and_grad(autoencoder_k, loss_fn_k)
    loss_and_grad_fn_v = nn.value_and_grad(autoencoder_v, loss_fn_v)
    
    epochs = 50
    print(f"\n[Trainer] Training Autoencoders via Context Distillation ({epochs} epochs)...")
    
    for epoch in range(epochs):
        total_loss = 0.0
        for k_island, v_island, k_summary, v_summary in dataset:
            loss_k, grads_k = loss_and_grad_fn_k(autoencoder_k, k_island, k_summary)
            optimizer_k.update(autoencoder_k, grads_k)
            mx.eval(autoencoder_k.parameters(), optimizer_k.state)
            
            loss_v, grads_v = loss_and_grad_fn_v(autoencoder_v, v_island, v_summary)
            optimizer_v.update(autoencoder_v, grads_v)
            mx.eval(autoencoder_v.parameters(), optimizer_v.state)
            
            total_loss += loss_k.item() + loss_v.item()
            
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:03d} | Loss: {total_loss / len(dataset):.6f}")

    print("\n[Trainer] Training Complete.")
    
    from mlx.utils import tree_flatten
    
    # Export Weights
    save_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
    os.makedirs(save_dir, exist_ok=True)
    
    weights = {}
    for k, v in tree_flatten(autoencoder_k.parameters()):
        weights[f"k_{k}"] = v
    for k, v in tree_flatten(autoencoder_v.parameters()):
        weights[f"v_{k}"] = v
        
    save_path = os.path.join(save_dir, "autoencoder_weights.safetensors")
    mx.save_safetensors(save_path, weights)
    print(f"[TSP] Weights successfully exported to {save_path}")

if __name__ == "__main__":
    train()