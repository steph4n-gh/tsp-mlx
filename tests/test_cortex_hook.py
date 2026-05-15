import mlx.core as mx
import pytest
from tsp_mlx.cortex_hook import CortexHook

def test_cortex_hook_initialization():
    hook = CortexHook(eval_interval=10, threshold=0.5, max_context_budget=4096)
    assert hook.base_interval == 10
    assert hook.threshold == 0.5
    assert hook.max_context_budget == 4096
    assert len(hook.edges) == 0

def test_cortex_hook_semantic_firewall():
    hook = CortexHook()
    hook.threat_indices = {5, 6}
    hook.execution_indices = {10}
    
    pos_ids = list(range(15))
    
    # Simulate an attention spike on both threat and execution (Dual Spike)
    # Decode phase creates a 1D attention mean
    attn_matrix = mx.zeros((1, 1, 1, 15))
    attn_matrix[0, 0, 0, 5] = 0.8  # Threat spike
    attn_matrix[0, 0, 0, 10] = 0.9 # Execution spike
    
    decision = hook.evaluate_attention(attn_matrix, sinks=[], position_ids=pos_ids)
    
    assert decision["action"] == "FATAL_BLOCK"

def test_cortex_hook_no_spike():
    hook = CortexHook()
    hook.threat_indices = {5, 6}
    hook.execution_indices = {10}
    
    pos_ids = list(range(15))
    
    # Simulate a safe attention pattern
    attn_matrix = mx.zeros((1, 1, 1, 15))
    attn_matrix[0, 0, 0, 1] = 0.8
    
    decision = hook.evaluate_attention(attn_matrix, sinks=[], position_ids=pos_ids)
    
    # Action should be ALLOW since we didn't hit eval_interval
    assert decision["action"] == "ALLOW"
