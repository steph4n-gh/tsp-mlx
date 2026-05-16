import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Module, r: int = 64, alpha: int = 32):
        super().__init__()
        self.linear = linear
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r
        
        # Handle both Linear and QuantizedLinear safely
        if hasattr(linear, "bits"):
            out_features, in_features_packed = linear.weight.shape
            in_features = in_features_packed * (32 // linear.bits)
            # Standard activation dtype for Apple Silicon quantized models
            self.target_dtype = mx.float16 
        else:
            out_features, in_features = linear.weight.shape
            self.target_dtype = getattr(linear.weight, "dtype", mx.float16)
        
        # 🛑 THE OPTIMIZER FIX: Strictly enforce FP32 initialization for LoRA weights.
        # If initialized in FP16, AdamW's g**2 variance tracking will overflow to infinity,
        # resulting in NaN parameters after the first optimization step.
        self.lora_a = mx.random.normal((in_features, r), dtype=mx.float32) * 1e-3
        self.lora_b = mx.zeros((r, out_features), dtype=mx.float32)
        
        # 🛑 THE FATAL COMPILER FIX:
        # Force the GPU to materialize the random tensors immediately. 
        # This prevents the LLVM compiler from choking on RNG nodes during FlashAttention.
        mx.eval(self.lora_a, self.lora_b)
        
        # 🛑 SPEED FIX: Bypass flag.
        self.is_active = True

    def __call__(self, x):
        if not self.is_active:
            return self.linear(x)
            
        # The LoRA math (x_32 @ FP32 @ FP32 prevents overflow in both directions)
        x_32 = x.astype(mx.float32)
        lora_res = (x_32 @ self.lora_a @ self.lora_b) * self.scale
        return self.linear(x) + lora_res.astype(self.target_dtype)
class MemoryConsolidator:
    def __init__(self, model: nn.Module = None, learning_rate: float = None, salience_threshold: float = 0.5):
        self.salience_threshold = salience_threshold
        self.model = model
        self.lora_layers = []
        
        # Dynamic Auto-Tuning Defaults
        self.learning_rate = 1e-5
        self.max_grad_norm = 10.0
        
        if model is not None:
            self._detect_dtype_and_tune(learning_rate)
            self._inject_lora()
            self.optimizer = optim.AdamW(learning_rate=self.learning_rate)
            self.load_adapters() # Automatically load persistent memory on boot

    def _detect_dtype_and_tune(self, user_lr):
        """Dynamically tunes hyperparameters based on model quantization volatility."""
        from .inference import find_layers
        layers = find_layers(self.model)
        if not layers: return
        
        # Check the first layer's value projection for quantization
        first_v_proj = None
        for layer in layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "v_proj"):
                first_v_proj = layer.self_attn.v_proj
                break
                
        if first_v_proj is not None:
            if isinstance(first_v_proj, LoRALinear):
                first_v_proj = first_v_proj.linear
            if hasattr(first_v_proj, "bits"):
                bits = first_v_proj.bits
                if bits <= 4:
                    self.learning_rate = 1e-6
                    self.max_grad_norm = 1.0
                    print(f"[TSP] \U0001F527 Dynamic Tuning: Detected {bits}-bit quantization. Setting LR={self.learning_rate}, Clip={self.max_grad_norm}")
                elif bits <= 8:
                    self.learning_rate = 5e-6
                    self.max_grad_norm = 5.0
                    print(f"[TSP] \U0001F527 Dynamic Tuning: Detected {bits}-bit quantization. Setting LR={self.learning_rate}, Clip={self.max_grad_norm}")
            else:
                self.learning_rate = 1e-5
                self.max_grad_norm = None
                print(f"[TSP] \U0001F527 Dynamic Tuning: Detected Unquantized weights. Setting LR={self.learning_rate}, Clip=None")
                
        if user_lr is not None:
            self.learning_rate = user_lr

    def _inject_lora(self):
        """Inject LoRA adapters into the value projection of ONLY the final transformer layer."""
        from .inference import find_layers
        layers = find_layers(self.model)
        if not layers:
            return

        self.model.freeze()

        # ONLY inject into the final layer to massively reduce TTT latency and memory overhead
        last_layer = layers[-1]

        # Most MLX models use 'v_proj'
        if hasattr(last_layer.self_attn, "v_proj"):
            orig_v_proj = last_layer.self_attn.v_proj
            if not isinstance(orig_v_proj, LoRALinear):
                lora_v = LoRALinear(orig_v_proj)
                last_layer.self_attn.v_proj = lora_v
                self.lora_layers.append(lora_v)
            else:
                self.lora_layers.append(orig_v_proj)

        for lora in self.lora_layers:
            lora.unfreeze()
            lora.linear.freeze()
    def evaluate_salience(self, attention_matrix: mx.array, island_physical_indices: list) -> float:
        if len(island_physical_indices) < 2:
            return 0.0
            
        # Salience = Mean attention density within the island
        # attention_matrix: [B, H, Lq, Lk]
        # We take the mean across heads and look at the island sub-matrix
        attn_mean = mx.mean(attention_matrix, axis=1)[0] # [Lq, Lk]
        
        island_arr = mx.array(island_physical_indices, dtype=mx.int32)
        island_attn = mx.take(mx.take(attn_mean, island_arr, axis=0), island_arr, axis=1)
        
        density = mx.mean(island_attn).item()
        
        # Scale density by log(size) to reward larger cohesive structures
        import math
        salience = density * math.log2(len(island_physical_indices))
        return salience

    def consolidate(self, x_island: mx.array, v_island: mx.array):
        """
        True Test-Time Training: Update LoRA weights to minimize reconstruction error
        of the island's Value vectors given its hidden states.
        """
        if not self.lora_layers:
            return

        print("\n[TSP] \U0001F9E0 MEMORY CONSOLIDATION TRIGGERED!")
        print("[TSP]   Performing True Test-Time Training (TTT) via LoRA...")
        
        with mx.stream(mx.gpu):
            # x_island: [B, L_island, D]
            # v_island: [B, n_heads, L_island, head_dim]
            # We need to reshape v_island to match the output of a linear layer [B, L, D]
            # where D = n_heads * head_dim
            B, n_heads, L, head_dim = v_island.shape
            v_target = v_island.transpose(0, 2, 1, 3).reshape(B, L, -1)
            
            # 🛑 CRITICAL FIX: DETACH TENSORS FROM THE MAIN GRAPH 
            x_island = mx.stop_gradient(x_island)
            v_target = mx.stop_gradient(v_target)
            
            # Materialize them immediately to prevent holding onto the entire hidden state graph
            mx.eval(x_island, v_target)
            
            if mx.any(mx.isnan(x_island)).item():
                print("[TSP] WARNING: x_island contains NaNs!")
            if mx.any(mx.isnan(v_target)).item():
                print("[TSP] WARNING: v_target contains NaNs!")
            
            # 🚀 OPTIMIZATION: Pre-calculate the output of the frozen linear layers once.
            # This avoids re-running 28+ large linear layers 3-5 times in the loop.
            frozen_outputs = []
            for lora in self.lora_layers:
                frozen = mx.stop_gradient(lora.linear(x_island))
                mx.eval(frozen)
                if mx.any(mx.isnan(frozen)).item():
                    print("[TSP] WARNING: frozen_outputs contains NaNs!")
                frozen_outputs.append(frozen)

            # 🛑 FIX: TTT Null Operation
            # The previous logic trained the LoRA to exactly 0.0 because v_target == frozen_outputs.
            # To perform actual Context Distillation, we must teach the LoRA to reconstruct the 
            # semantic average of the evicted values when given the Macro-Token's hidden state.
            macro_x = mx.mean(x_island, axis=1, keepdims=True)
            macro_v_target = mx.mean(v_target, axis=1, keepdims=True)
            
            macro_x = mx.stop_gradient(macro_x)
            macro_v_target = mx.stop_gradient(macro_v_target)
            mx.eval(macro_x, macro_v_target)

            def loss_fn(model_params):
                total_loss = 0
                for i, lora in enumerate(self.lora_layers):
                    # Pass the Macro-Token hidden state through the frozen projection
                    frozen_macro_v = mx.stop_gradient(lora.linear(macro_x)).astype(mx.float32)
                    
                    # Pass the Macro-Token hidden state through the LoRA adapter
                    lora_res = (macro_x.astype(mx.float32) @ lora.lora_a @ lora.lora_b) * lora.scale
                    
                    # The prediction is the combined output
                    v_pred = frozen_macro_v + lora_res
                    
                    # The loss is how well the combined output matches the TRUE average of the evicted sequence
                    total_loss += mx.mean(mx.square(v_pred - macro_v_target.astype(mx.float32)))
                return total_loss / len(self.lora_layers)

            loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
            
            # Perform 10 steps of optimization for deeper memorization
            for step in range(10):
                loss, grads = loss_and_grad_fn(self.model)
                
                self.optimizer.update(self.model, grads)
                mx.eval(self.model.trainable_parameters(), self.optimizer.state)
                if step == 0:
                    print(f"[TSP]   Initial TTT Loss: {loss.item():.6f}")

            final_loss = loss.item()
            self.last_ttt_loss = final_loss
            print(f"[TSP]   Final TTT Loss: {final_loss:.6f}")
            print("[TSP]   Semantic manifold updated. Resuming generation.")
            
            # Activate the LoRA path now that weights have been updated
            for lora in self.lora_layers:
                lora.is_active = True
            
            # Final cleanup of the TTT graph
            mx.clear_cache()

    def save_adapters(self, path="tsp_adapters.safetensors"):
        tensors = {}
        for i, lora in enumerate(self.lora_layers):
            tensors[f"layer_{i}.lora_a"] = lora.lora_a
            tensors[f"layer_{i}.lora_b"] = lora.lora_b
        mx.save_safetensors(path, tensors)
        print(f"[TSP]   Adapters saved to {path} (Permanent Learning).")

    def load_adapters(self, path="tsp_adapters.safetensors"):
        import os
        if not os.path.exists(path): return
        try:
            tensors = mx.load(path)
            for i, lora in enumerate(self.lora_layers):
                if f"layer_{i}.lora_a" in tensors:
                    lora.lora_a = tensors[f"layer_{i}.lora_a"]
                if f"layer_{i}.lora_b" in tensors:
                    lora.lora_b = tensors[f"layer_{i}.lora_b"]
                lora.is_active = True
            print(f"[TSP] \U0001F4BE Loaded persistent learning adapters from {path}")
        except Exception as e:
            print(f"[TSP] \U0001F6A8 Failed to load adapters: {e}")
