import os
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from tsp_mlx.compression import SubManifoldAutoencoder
import json
import argparse

def generate_training_data(model, tokenizer, num_samples=5, dataset_name=None):
    """
    Generates (island_kv, summary_kv) pairs for Distillation training.
    """
    samples = []
    if dataset_name:
        try:
            print(f"[Trainer] Loading large-scale dataset: {dataset_name}...")
            from datasets import load_dataset
            # Load a small slice for memory safety, but can be scaled to the full dataset
            dataset = load_dataset(dataset_name, split='train', streaming=True)
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                # Usually text is in 'text' or 'content' key
                text = item.get('text', item.get('content', str(item)))
                if len(text) > 100: # Ensure it's long enough to be an 'island'
                    # truncate to a reasonable island size (e.g. 500 chars)
                    samples.append(text[:500])
        except ImportError:
            print("[Trainer] WARNING: 'datasets' library not found. Falling back to synthetic data.")
            print("[Trainer] Run `pip install datasets` to train on real-world Hugging Face datasets.")
            dataset_name = None
            
    if not dataset_name or not samples:
        print(f"[Trainer] Generating {num_samples} synthetic distillation samples...")
        synthetic = [
            "The quick brown fox jumps over the lazy dog. This is a classic pangram used to test typewriters and keyboards.",
            "Quantum entanglement is a physical phenomenon that occurs when a group of particles are generated, interact, or share spatial proximity in a way such that the quantum state of each particle of the group cannot be described independently of the state of the others.",
            "To make a Margherita pizza, you need dough, San Marzano tomatoes, mozzarella cheese, fresh basil, salt, and extra-virgin olive oil.",
            "Isaac Newton's laws of motion are three basic laws of classical mechanics that describe the relationship between the motion of an object and the forces acting on it.",
            "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation."
        ]
        # Repeat synthetic data if num_samples is large
        while len(samples) < num_samples:
            samples.extend(synthetic)
        samples = samples[:num_samples]
    
    training_pairs = []
    
    for idx, text in enumerate(samples):
        if idx > 0 and idx % 100 == 0:
            print(f"  ...processed {idx}/{num_samples} samples")
            
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
        k_summary = summary_caches[0].keys[:, :, -1:, :]
        v_summary = summary_caches[0].values[:, :, -1:, :]
        
        training_pairs.append((k_island, v_island, k_summary, v_summary))
        
    return training_pairs

def train(args):
    print(f"[Trainer] Loading model ({args.model})...")
    model, tokenizer = load(args.model)
    
    # Check head_dim
    dummy_cache = make_prompt_cache(model)
    _ = model(mx.array([[0]]), cache=dummy_cache)
    head_dim = dummy_cache[0].keys.shape[-1]
    
    print(f"[Trainer] Detected KV Head Dim: {head_dim}")
    
    autoencoder_k = SubManifoldAutoencoder(hidden_dim=head_dim)
    autoencoder_v = SubManifoldAutoencoder(hidden_dim=head_dim)
    
    mx.eval(autoencoder_k.parameters())
    mx.eval(autoencoder_v.parameters())
    
    optimizer_k = optim.AdamW(learning_rate=args.lr)
    optimizer_v = optim.AdamW(learning_rate=args.lr)
    
    dataset = generate_training_data(model, tokenizer, num_samples=args.samples, dataset_name=args.dataset)
    
    def loss_fn_k(model_k, x, y):
        pred = model_k(x)
        return mx.mean(mx.square(pred - y))
        
    def loss_fn_v(model_v, x, y):
        pred = model_v(x)
        return mx.mean(mx.square(pred - y))
        
    loss_and_grad_fn_k = nn.value_and_grad(autoencoder_k, loss_fn_k)
    loss_and_grad_fn_v = nn.value_and_grad(autoencoder_v, loss_fn_v)
    
    print(f"\n[Trainer] Training Autoencoders via Context Distillation ({args.epochs} epochs)...")
    
    for epoch in range(args.epochs):
        total_loss = 0.0
        for k_island, v_island, k_summary, v_summary in dataset:
            loss_k, grads_k = loss_and_grad_fn_k(autoencoder_k, k_island, k_summary)
            optimizer_k.update(autoencoder_k, grads_k)
            mx.eval(autoencoder_k.parameters(), optimizer_k.state)
            
            loss_v, grads_v = loss_and_grad_fn_v(autoencoder_v, v_island, v_summary)
            optimizer_v.update(autoencoder_v, grads_v)
            mx.eval(autoencoder_v.parameters(), optimizer_v.state)
            
            total_loss += loss_k.item() + loss_v.item()
            
        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
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
    parser = argparse.ArgumentParser(description="Train SubManifold Autoencoders for TSP.")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen2.5-0.5B-Instruct-4bit", help="Base model to distill KV caches from.")
    parser.add_argument("--dataset", type=str, default=None, help="Hugging Face dataset repo (e.g., 'Salesforce/longbench').")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples/islands to process.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for AdamW.")
    args = parser.parse_args()
    
    train(args)