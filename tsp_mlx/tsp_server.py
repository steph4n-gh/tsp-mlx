from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlx.core as mx
from mlx_lm import load
from tsp_mlx.sparse_cache import KVCacheManager
from tsp_mlx.cortex_hook import CortexHook
from tsp_mlx.inference import generate_infinite_context
from fastapi.responses import StreamingResponse
import time

app = FastAPI(title="\u03C4-Spectral Pruner API Harness")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model = None
tokenizer = None

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "default"
    messages: list[Message]
    stream: bool = False
    max_tokens: int = 1024
    temperature: float = 0.7

@app.get("/v1/models")
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

@app.get("/api/tags")
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

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    global model, tokenizer
    if not model or not tokenizer:
        return {"error": "Model not loaded properly."}

    messages_dict = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt = tokenizer.apply_chat_template(messages_dict, tokenize=False, add_generation_prompt=True)
    input_ids = mx.array(tokenizer.encode(prompt))[None]
    
    print(f"\n[API] Received request: {len(input_ids[0])} context tokens.")
    
    seq_len = input_ids.shape[1]
    manager = model.tsp_kv_manager
    if not hasattr(manager.position_tracker, '_positions') or manager.position_tracker._positions.shape[0] == 0:
        manager.position_tracker.step(seq_len)
    else:
        manager.position_tracker._positions = mx.array([], dtype=mx.int32)
        manager.position_tracker.current_pos = 0
        manager.position_tracker.step(seq_len)

    generator = generate_infinite_context(model, input_ids, max_tokens=req.max_tokens)
    
    def stream_tokens():
        for token, stats in generator:
            if token.item() == tokenizer.eos_token_id:
                break
            text = tokenizer.decode([token.item()])
            yield f"data: {{\"choices\": [{{\"delta\": {{\"content\": \"{text}\"}}}}]}}\n\n"
        yield "data: [DONE]\n\n"

    if req.stream:
        return StreamingResponse(stream_tokens(), media_type="text/event-stream")
    else:
        response_text = ""
        for token, stats in generator:
            if token.item() == tokenizer.eos_token_id:
                break
            response_text += tokenizer.decode([token.item()])
        return {
            "id": "chatcmpl-tsp",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(input_ids[0]), "completion_tokens": len(tokenizer.encode(response_text)), "total_tokens": len(input_ids[0]) + len(tokenizer.encode(response_text))}
        }

def run_server():
    global model, tokenizer
    import uvicorn
    
    model_name = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    print(f"Booting TSP API Harness ({model_name})...")
    try:
        model, tokenizer = load(model_name)
        hook = CortexHook(eval_interval=10, threshold=0.9)
        manager = KVCacheManager(hook, model=model, enable_compression=True, enable_consolidation=True)
        manager.consolidator.salience_threshold = 0.0 
        model.tsp_kv_manager = manager
        print("\033[1;32m[SUCCESS] Model loaded and TSP Hook attached.\033[0m")
    except Exception as e:
        print(f"\033[1;31m[ERROR] Failed to load model: {e}\033[0m")

    print("\033[1;36mStarting TSP Local API Server on http://127.0.0.1:8080\033[0m")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")

if __name__ == "__main__":
    run_server()
