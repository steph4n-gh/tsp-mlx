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

API_KEY = secrets.token_hex(16)
logger.info(f"============================================================")
logger.info(f" \U0001F512 SECURE API KEY GENERATED: {API_KEY}")
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
    global model, tokenizer
    if not model or not tokenizer:
        return {"error": "Model not loaded properly."}

    # Extract untrusted indices
    untrusted_indices = set()
    current_idx = 0
    messages_dict = []
    
    for m in req.messages:
        msg_dict = {"role": m.role, "content": m.content}
        messages_dict.append(msg_dict)
        # We need to approximate the token indices for this message.
        # A full proper implementation would encode message by message, but for the prototype:
        msg_tokens = tokenizer.encode(m.content)
        if m.is_untrusted:
            for i in range(len(msg_tokens)):
                # Roughly offset by current_idx and role tokens
                untrusted_indices.add(current_idx + i + 4) 
        current_idx += len(msg_tokens) + 4 # Rough header size

    prompt = tokenizer.apply_chat_template(messages_dict, tokenize=False, add_generation_prompt=True)
    input_ids = mx.array(tokenizer.encode(prompt))[None]
    
    logger.info(f"Received request: {len(input_ids[0])} context tokens.")
    
    seq_len = input_ids.shape[1]
    
    from tsp_mlx.generate import generate_with_tsp
    
    # We still need head_dim for setup
    from mlx_lm.models.cache import make_prompt_cache
    dummy_cache = make_prompt_cache(model)
    head_dim = dummy_cache[0].keys.shape[-1]
    for c in dummy_cache: c.keys = None; c.values = None
    del dummy_cache

    generator = generate_with_tsp(model, tokenizer, prompt, max_tokens=req.max_tokens, head_dim=head_dim, untrusted_indices=untrusted_indices)

    async def stream_tokens():
        nonlocal generator
        try:
            async for text, stats in generator:
                safe_stats = {
                    "active_positions_count": len(stats.get("active_positions", [])),
                    "total_evicted": stats.get("total_evicted", 0),
                    "lambda_2": stats.get("lambda_2", 0.0)
                }
                
                data = {
                    "choices": [{"delta": {"content": text}}],
                    "tsp_stats": safe_stats
                }
                yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            generator = None

    if req.stream:
        return StreamingResponse(stream_tokens(), media_type="text/event-stream")
    else:
        try:
            response_text = ""
            async for text, stats in generator:
                response_text += text
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
    
    model_name = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
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
