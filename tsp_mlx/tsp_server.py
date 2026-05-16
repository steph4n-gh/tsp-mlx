import secrets
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlx.core as mx
from mlx_lm import load
from tsp_mlx.sparse_cache import KVCacheManager
from tsp_mlx.cortex_hook import CortexHook
from tsp_mlx.inference import generate_infinite_context
from fastapi.responses import StreamingResponse
import time
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tsp_server")

API_KEY = "tsp-dev-key"
logger.info(f"============================================================")
logger.info(f" \U0001F512 SECURE API KEY: {API_KEY}")
logger.info(f"============================================================")

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return True

app = FastAPI(title="\u03C4-Spectral Pruner API Harness")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global state
model = None
tokenizer = None
global_kv_manager = None
global_kv_caches = None
global_prompt = ""

class Message(BaseModel):
    role: str
    content: str
    is_untrusted: bool = False

class ChatRequest(BaseModel):
    model: str = "default"
    messages: list[Message]
    stream: bool = False
    max_tokens: int = 1024
    temperature: float = 0.7

@app.get("/v1/models", dependencies=[Depends(verify_token)])
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "tsp-qwen-7b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "tsp"
            }
        ]
    }

@app.get("/api/tags", dependencies=[Depends(verify_token)])
async def ollama_tags():
    return {
        "models": [
            {
                "name": "tsp-qwen-7b",
                "modified_at": "2026-05-14T10:00:00.000Z",
                "size": 4000000000,
                "digest": "sha256:1234567890abcdef",
                "details": {
                    "format": "gguf",
                    "family": "qwen2",
                    "families": ["qwen2"],
                    "parameter_size": "7B",
                    "quantization_level": "Q4_0"
                }
            }
        ]
    }

@app.post("/v1/chat/completions", dependencies=[Depends(verify_token)])
async def chat_completions(req: ChatRequest):
    global model, tokenizer, global_kv_manager, global_kv_caches, global_prompt
    if not model or not tokenizer:
        return {"error": "Model not loaded properly."}

    messages_dict = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt = tokenizer.apply_chat_template(messages_dict, tokenize=False, add_generation_prompt=True)
    
    # 🛑 STATEFUL CACHE LOGIC: Diff the prompt to only process new tokens
    is_continuation = False
    if global_kv_caches is not None and prompt.startswith(global_prompt) and len(global_prompt) > 0:
        new_string = prompt[len(global_prompt):]
        input_ids = mx.array(tokenizer.encode(new_string))[None]
        is_continuation = True
        logger.info(f"Continuing session. Sliced {len(input_ids[0])} new tokens.")
    else:
        # Reset session
        input_ids = mx.array(tokenizer.encode(prompt))[None]
        
        from tsp_mlx.generate import setup_tsp
        from mlx_lm.models.cache import make_prompt_cache
        
        global_kv_manager = setup_tsp(model, head_dim=128, enable_compression=True, enable_consolidation=True)
        global_kv_manager.consolidator.salience_threshold = 0.0
        global_kv_caches = make_prompt_cache(model)
        
        # Initial dummy evaluate to prepare cache
        dummy_logits = model(mx.array([[0]]), cache=global_kv_caches)
        mx.eval(dummy_logits)
        for c in global_kv_caches:
            c.keys = None
            c.values = None
            
        logger.info(f"New session started. Prefilling {len(input_ids[0])} tokens.")
        
    seq_len = input_ids.shape[1]
    
    # Run the prefill for the new tokens
    prefill_ids = input_ids[:, :-1] if is_continuation else input_ids[:, :-1]
    # Wait, if is_continuation, input_ids contains just the new tokens. 
    # Actually, input_ids[:, :-1] works for both, except if input_ids is length 1.
    if prefill_ids.shape[1] > 0:
        global_kv_manager.position_tracker.step(prefill_ids.shape[1])
        if hasattr(model, "model"):
            _ = model.model(prefill_ids, cache=global_kv_caches)
        else:
            _ = model(prefill_ids, cache=global_kv_caches)
            
        cache_tensors = []
        for c in global_kv_caches:
            if c.keys is not None: cache_tensors.append(c.keys)
            if c.values is not None: cache_tensors.append(c.values)
            if hasattr(c, 'x_states') and c.x_states is not None: cache_tensors.append(c.x_states)
        mx.eval(*cache_tensors)
        mx.synchronize()

    # Create the generator for the final token
    final_input = input_ids[:, -1:] if input_ids.shape[1] > 0 else mx.array([[tokenizer.eos_token_id]], dtype=mx.int32)
    
    generator = generate_infinite_context(
        model, 
        final_input, 
        max_tokens=req.max_tokens, 
        kv_manager=global_kv_manager, 
        temp=req.temperature,
        repetition_penalty=1.1,
        kv_caches=global_kv_caches
    )

    async def stream_tokens():
        nonlocal generator
        global global_prompt
        try:
            full_response = ""
            async for token, stats in generator:
                token_id = token.item()
                if token_id == tokenizer.eos_token_id:
                    break
                text = tokenizer.decode([token_id], skip_special_tokens=True)
                
                full_response += text
                safe_stats = {
                    "active_positions_count": len(stats.get("active_positions", [])),
                    "total_evicted": stats.get("total_evicted", 0),
                    "lambda_2": stats.get("lambda_2", 0.0),
                    "macro_tokens": stats.get("macro_tokens", 0),
                    "last_ttt_loss": stats.get("last_ttt_loss", 0.0),
                    "max_context_budget": stats.get("max_context_budget", 2048)
                }
                
                data = {
                    "choices": [{"delta": {"content": text}}],
                    "tsp_stats": safe_stats
                }
                yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"
            global_prompt = prompt + full_response
        finally:
            generator = None

    if req.stream:
        return StreamingResponse(stream_tokens(), media_type="text/event-stream")
    else:
        try:
            response_text = ""
            async for token, stats in generator:
                token_id = token.item()
                if token_id == tokenizer.eos_token_id:
                    break
                response_text += tokenizer.decode([token_id], skip_special_tokens=True)
            global_prompt = prompt + response_text
            return {
                "id": "chatcmpl-tsp",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(input_ids[0]), "completion_tokens": len(tokenizer.encode(response_text)), "total_tokens": len(input_ids[0]) + len(tokenizer.encode(response_text))}
            }
        finally:
            generator = None

def run_server():
    global model, tokenizer
    import uvicorn
    
    model_name = "mlx-community/Qwen2.5-Coder-7B-Instruct-8bit"
    logger.info(f"Booting TSP API Harness ({model_name})...")
    try:
        from tsp_mlx.inference import patch_attention_for_extraction, patch_rope_for_sparse_positions
        model, tokenizer = load(model_name)
        hook = CortexHook(eval_interval=10, threshold=0.9)
        manager = KVCacheManager(hook, model=model, enable_compression=True, enable_consolidation=True)
        manager.consolidator.salience_threshold = 0.0 
        manager.consolidator.load_adapters()
        model._tsp_kv_manager = manager
        patch_attention_for_extraction(model)
        patch_rope_for_sparse_positions(model, manager.position_tracker)
        logger.info("Model loaded and TSP Hook attached.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

    logger.info("Starting TSP Local API Server on http://127.0.0.1:8080")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")

if __name__ == "__main__":
    run_server()
