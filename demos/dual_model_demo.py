import sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from tsp_mlx.sparse_cache import KVCacheManager
from tsp_mlx.cortex_hook import CortexHook
from tsp_mlx.inference import patch_rope_for_sparse_positions, patch_attention_for_extraction
import time

def clear_screen():
    print("\033[2J\033[H", end="")

class Agent:
    def __init__(self, name, model_path, system_prompt):
        self.name = name
        self.model_path = model_path
        print(f"Loading {name} ({model_path})...")
        self.model, self.tokenizer = load(model_path)
        
        dummy_cache = make_prompt_cache(self.model)
        _ = self.model(mx.array([[0]]), cache=dummy_cache)
        head_dim = dummy_cache[0].keys.shape[-1]
        
        # Adaptive threshold: starts at 0.05, auto-scales if context approaches max_context_budget
        # Force a smaller max budget for the demo to guarantee aggressive pruning and flat VRAM
        self.hook = CortexHook(eval_interval=10, threshold=0.05, max_context_budget=200)
        
        # Enable compression to test the new deterministic SVD/Variance compressor!
        self.manager = KVCacheManager(self.hook, model=self.model, enable_compression=True, enable_consolidation=True, head_dim=head_dim)
        self.manager.consolidator.salience_threshold = 0.0 # Force TTT for demo
        self.model._tsp_kv_manager = self.manager
        
        patch_rope_for_sparse_positions(self.model, self.manager.position_tracker)
        patch_attention_for_extraction(self.model)
        
        self.kv_caches = make_prompt_cache(self.model)
        
        if "gemma" in model_path.lower():
            self.start_tag = "<start_of_turn>"
            self.end_tag = "<end_of_turn>"
            self.sys_role = "user" # Gemma doesn't natively support system role, uses user
            self.asst_role = "model"
            self.stop_tokens = [self.tokenizer.eos_token_id, self.tokenizer.encode("<end_of_turn>")[0]]
        else:
            self.start_tag = "<|im_start|>"
            self.end_tag = "<|im_end|>"
            self.sys_role = "system"
            self.asst_role = "assistant"
            self.stop_tokens = [self.tokenizer.eos_token_id, self.tokenizer.encode("<|im_end|>")[0]]
        
        # Inject system prompt
        prompt = f"{self.start_tag}{self.sys_role}\n{system_prompt}{self.end_tag}\n"
        ids = mx.array(self.tokenizer.encode(prompt))[None]
        self.manager.position_tracker.step(ids.shape[1])
        _ = self.model(ids, cache=self.kv_caches)
        
        # Evaluate to seal the graph
        mx.eval(*[c.keys for c in self.kv_caches], *[c.values for c in self.kv_caches])
        
        # Protect system prompt
        self.sink_tokens = list(range(ids.shape[1]))
        self.total_gen = ids.shape[1]
        self.last_evicted = 0
        self.action_log = ""
        self.next_token = None

    def hear(self, text, sender_name):
        prompt = f"{self.start_tag}user\n[{sender_name}]: {text}{self.end_tag}\n{self.start_tag}{self.asst_role}\n"
        ids = mx.array(self.tokenizer.encode(prompt))[None]
        self.manager.position_tracker.step(ids.shape[1])
        logits = self.model(ids, cache=self.kv_caches)
        self.total_gen += ids.shape[1]
        
        # TSP Update for Prefill
        if hasattr(self.manager, 'last_attention_matrix'):
            attn_matrix = self.manager.last_attention_matrix
            raw_caches = [(c.keys, c.values) for c in self.kv_caches]
            before_len = self.manager.position_tracker.get_positions().shape[0]
            
            pruned = self.manager.update(attn_matrix, raw_caches, sinks=self.sink_tokens, x=None)
            
            if pruned is not raw_caches:
                for cache_obj, (pk, pv) in zip(self.kv_caches, pruned):
                    cache_obj.keys = pk
                    cache_obj.values = pv
                    cache_obj.offset = pk.shape[2]
            
            after_len = self.manager.position_tracker.get_positions().shape[0]
            evicted = before_len - after_len
            if evicted > 0:
                self.last_evicted += evicted
                self.action_log = f"\u26A0\uFE0F {self.name} Pruned {evicted} dead tokens during prefill."
        
        # Evaluate to prevent memory leaks from the prefill phase
        eval_list = [logits, self.manager.position_tracker.get_positions()]
        for c in self.kv_caches:
            eval_list.extend([c.keys, c.values])
        mx.eval(*eval_list)
        mx.metal.clear_cache()
        
        # We must divide by temp here if we want temp sampling, or just argmax
        temp = 0.7
        logits_step = logits[:, -1, :] / temp
        self.next_token = mx.random.categorical(logits_step, num_samples=1)

    def speak(self, max_tokens=100):
        if self.next_token is None:
            return ""
            
        y = self.next_token
        response = ""
        
        for i in range(max_tokens):
            self.manager.position_tracker.step(1)
            logits = self.model(y, cache=self.kv_caches)
            
            # Temperature sampling to prevent repetitive loops
            temp = 0.7
            logits_step = logits[:, -1, :] / temp
            y = mx.random.categorical(logits_step, num_samples=1)
            token_id = y.item()
            
            if token_id in self.stop_tokens:
                break
                
            word = self.tokenizer.decode([token_id])
            response += word
            self.total_gen += 1
            
            # Print word by word
            print(word, end="", flush=True)
            
            # TSP Update
            if hasattr(self.manager, 'last_attention_matrix'):
                attn_matrix = self.manager.last_attention_matrix
                raw_caches = [(c.keys, c.values) for c in self.kv_caches]
                before_len = self.manager.position_tracker.get_positions().shape[0]
                
                pruned = self.manager.update(attn_matrix, raw_caches, sinks=self.sink_tokens, x=None)
                
                if pruned is not raw_caches:
                    for cache_obj, (pk, pv) in zip(self.kv_caches, pruned):
                        cache_obj.keys = pk
                        cache_obj.values = pv
                        cache_obj.offset = pk.shape[2]
                
                # CRITICAL: mx.eval EVERYTHING to clear the graph and prevent memory leaks
                eval_list = [y, self.manager.position_tracker.get_positions()]
                for c in self.kv_caches:
                    eval_list.extend([c.keys, c.values])
                mx.eval(*eval_list)
                mx.metal.clear_cache()
                
                after_len = self.manager.position_tracker.get_positions().shape[0]
                evicted = before_len - after_len
                if evicted > 0:
                    self.last_evicted += evicted
                    self.action_log = f"\u26A0\uFE0F {self.name} Pruned {evicted} dead tokens."
                    
        response = response.strip()
        response = response.replace(f"[{self.name}]:", "").strip()
        return response

def print_dashboard(agent_a, agent_b, turn):
    clear_screen()
    print(f"\033[1;37m[\u03C4-Spectral Pruner] DUAL-MODEL NEURAL CHAT (Turn {turn})\033[0m")
    print("\033[1;30m-------------------------------------------------------------\033[0m")
    
    print("\n\033[1;34m[AGENT ALPHA: Qwen 7B Coder]\033[0m")
    print(f"  \u25B6 Active VRAM Tokens: {len(agent_a.manager.position_tracker.position_ids)}")
    print(f"  \u25B6 Total Evicted:      {agent_a.last_evicted}")
    if agent_a.action_log: print(f"  {agent_a.action_log}")
    
    print("\n\033[1;32m[AGENT BETA: Gemma-2 2B]\033[0m")
    print(f"  \u25B6 Active VRAM Tokens: {len(agent_b.manager.position_tracker.position_ids)}")
    print(f"  \u25B6 Total Evicted:      {agent_b.last_evicted}")
    if agent_b.action_log: print(f"  {agent_b.action_log}")
    
    print("\033[1;30m-------------------------------------------------------------\033[0m")
    print("\n\033[1;37mLIVE TRANSCRIPT:\033[0m")

def main():
    agent_a = Agent("Agent Alpha", "mlx-community/Qwen2.5-Coder-7B-Instruct-8bit", "You are an expert, highly logical Software Architect. Keep responses under 3 sentences.")
    agent_b = Agent("Agent Beta", "mlx-community/gemma-2-2b-it-4bit", "You are an imaginative, philosophical AI. You question the nature of code and logic. Keep responses under 3 sentences.")
    
    chat_history = []
    
    seed_prompt = "I believe code is simply deterministic logic. There is no room for creativity in pure software engineering. What do you think?"
    
    print_dashboard(agent_a, agent_b, 0)
    print(f"\033[1;34m[Agent Alpha]: {seed_prompt}\033[0m\n")
    chat_history.append(( "\033[1;34m", f"[Agent Alpha]: {seed_prompt}" ))
    
    current_msg = seed_prompt
    
    for turn in range(1, 50):
        print_dashboard(agent_a, agent_b, turn)
        for color, text in chat_history[-4:]:
            print(f"{color}{text}\033[0m\n")
            
        print("\033[1;32m[Agent Beta]: \033[0m", end="", flush=True)
        agent_b.hear(current_msg, "Agent Alpha")
        beta_resp = agent_b.speak(max_tokens=500)
        chat_history.append(( "\033[1;32m", f"[Agent Beta]: {beta_resp}" ))
        print("\n")
        
        print_dashboard(agent_a, agent_b, turn)
        for color, text in chat_history[-4:]:
            print(f"{color}{text}\033[0m\n")
            
        print("\033[1;34m[Agent Alpha]: \033[0m", end="", flush=True)
        agent_a.hear(beta_resp, "Agent Beta")
        alpha_resp = agent_a.speak(max_tokens=500)
        chat_history.append(( "\033[1;34m", f"[Agent Alpha]: {alpha_resp}" ))
        print("\n")
        
        current_msg = alpha_resp

    print("\n\n[TSP] Dual-Model Chat Complete.")

if __name__ == "__main__":
    main()
