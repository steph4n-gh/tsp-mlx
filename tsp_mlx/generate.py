import mlx.core as mx
from tsp_mlx.sparse_cache import KVCacheManager
from tsp_mlx.cortex_hook import CortexHook
from tsp_mlx.inference import patch_attention_for_extraction, patch_rope_for_sparse_positions, generate_infinite_context
from mlx_lm.models.cache import make_prompt_cache

def setup_tsp(model, head_dim: int = 128, enable_compression: bool = True, enable_consolidation: bool = True, untrusted_indices: set = None):
    """
    Abstracts the setup of the TSP engine for an MLX model.
    """
    hook = CortexHook(eval_interval=10, threshold=0.9)
    manager = KVCacheManager(hook, model=model, enable_compression=enable_compression, enable_consolidation=enable_consolidation, head_dim=head_dim)
    
    if untrusted_indices:
        manager.untrusted_indices = untrusted_indices
        
    if enable_consolidation:
        manager.consolidator.salience_threshold = 0.0
        manager.consolidator.load_adapters()
        
    model._tsp_kv_manager = manager
    patch_attention_for_extraction(model)
    patch_rope_for_sparse_positions(model, manager.position_tracker)
    return manager

async def generate_with_tsp(model, tokenizer, prompt: str, max_tokens: int = 100, head_dim: int = 128, untrusted_indices: set = None, temp: float = 0.0, **kwargs):
    """
    A high-level wrapper for generation that handles all TSP loop complexity.
    Yields (text_chunk, stats_dict).
    """
    manager = setup_tsp(model, head_dim=head_dim, untrusted_indices=untrusted_indices)
        
    input_ids = mx.array(tokenizer.encode(prompt))[None]
    
    # Run the initial prefill
    dummy_cache = make_prompt_cache(model)
    dummy_logits = model(mx.array([[0]]), cache=dummy_cache)
    mx.eval(dummy_logits)
    for c in dummy_cache:
        c.keys = None
        c.values = None
    del dummy_logits, dummy_cache
    
    mx.synchronize()
    mx.eval(model.parameters())
    
    generator = generate_infinite_context(model, input_ids, max_tokens=max_tokens, kv_manager=manager, temp=temp)
    
    try:
        async for token, stats in generator:
            token_id = token.item()
            if token_id == tokenizer.eos_token_id:
                break
            text = tokenizer.decode([token_id])
            yield text, stats
    finally:
        # Guarantee rigorous Garbage Collection
        if hasattr(model, "_tsp_kv_manager") and model._tsp_kv_manager:
            model._tsp_kv_manager.last_attention_matrix = None
            model._tsp_kv_manager.last_hidden_states = None
        model._tsp_kv_manager = None
        manager = None
        generator = None
        import gc
        mx.clear_cache()
        gc.collect()
