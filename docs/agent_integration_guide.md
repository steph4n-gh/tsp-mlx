# Integrating τ-Spectral Pruner (TSP) with Autonomous Agents

TSP is designed to operate seamlessly underneath the application layer. Because it hijacks the inference engine (MLX) directly, the parent agent framework usually has no idea its memory is being actively managed and pruned.

This guide outlines the Integration Matrix and the two primary approaches to injecting TSP into popular Autonomous Agent workflows.

---

## The Integration Matrix

Different AI tools interact with local models in fundamentally different ways. Use this matrix to determine the correct integration approach for your workflow:

| Tool / Framework | Architecture Type | TSP Integration Method | Setup Difficulty |
| :--- | :--- | :--- | :--- |
| **Cursor (IDE)** | API Client (HTTP) | Local API Server | Easy |
| **Continue.dev** | API Client (HTTP) | Local API Server | Easy |
| **Chatbox / LM Studio** | API Client (HTTP) | Local API Server | Easy |
| **Gemini CLI** | Python App (Direct MLX) | MLX Monkey Patch | Medium |
| **Aider** | Python App / LiteLLM | Local API Server or Patch | Medium |
| **OpenCode CLI** | Python App / Executable | CLI Wrapper / Patch | Hard |
| **Claude Code (OSS Clones)** | Executable Binary | CLI Wrapper / Patch | Hard |

### Rule of Thumb:
*   **If the tool asks for a "Base URL" or "OpenAI Endpoint":** Use the **Local API Server** method.
*   **If the tool is a Python script that loads MLX directly:** Use the **MLX Monkey Patch** method.
*   **If the tool is a standalone CLI binary:** You must wrap the binary execution in a shell script that injects the Python patch.

---

## Approach 1: The Local API Server (For API Clients like Cursor, Continue.dev)

Applications like **Cursor** or **Continue.dev** format massive text prompts and send them to an API endpoint (usually `api.openai.com`).

To give these tools infinite memory, we build a local "TSP Harness"—a lightweight HTTP server that pretends to be OpenAI, but actually routes the prompt through our local MLX model equipped with TSP.

### The Detailed API Integration Tutorial

#### Step 1: Install FastAPI
We need a lightweight web framework to handle the HTTP requests.
```bash
pip install fastapi uvicorn
```

#### Step 2: Build the TSP Harness Server (`tsp_server.py`)
Create a Python file that exposes an `/v1/chat/completions` endpoint.

```python
from fastapi import FastAPI
from pydantic import BaseModel
import mlx.core as mx
from mlx_lm import load
from tsp_mlx.sparse_cache import KVCacheManager
from tsp_mlx.cortex_hook import CortexHook
from tsp_mlx.inference import generate_infinite_context
from fastapi.responses import StreamingResponse

app = FastAPI()

# 1. Load the model globally when the server boots
print("Booting TSP API Harness...")
model, tokenizer = load("mlx-community/Qwen2.5-Coder-14B-Instruct-4bit")

# 2. Attach the TSP Brain
hook = CortexHook(eval_interval=10, threshold=0.1)
model.tsp_kv_manager = KVCacheManager(hook, model=model, enable_compression=True, enable_consolidation=True)

class ChatRequest(BaseModel):
    messages: list
    stream: bool = False

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    prompt = tokenizer.apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True)
    input_ids = mx.array(tokenizer.encode(prompt))[None]
    
    generator = generate_infinite_context(model, input_ids, max_tokens=1024)
    
    def stream_tokens():
        for token, stats in generator:
            if token.item() == tokenizer.eos_token_id:
                break
            text = tokenizer.decode([token.item()])
            yield f"data: {{\"choices\": [{{\"delta\": {{\"content\": \"{text}\"}}}}]}}\n\n"
        yield "data: [DONE]\n\n"

    if req.stream:
        return StreamingResponse(stream_tokens(), media_type="text/event-stream")
    return {"choices": [{"message": {"content": "Non-streaming not fully implemented"}}]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
```

#### Step 3: Wire up your IDE / Agent
1. Open your IDE's AI settings (e.g., Cursor or Continue.dev).
2. Set the provider to **OpenAI Compatible**.
3. Set the **Base URL** to: `http://127.0.0.1:8080/v1`
4. Set the **API Key** to any dummy value.
5. You can now dump entire repositories into the chat interface. Your local server will swallow the prompt, prune irrelevant files instantly, and stream the generated code back to the UI.

---

## Approach 2: The MLX Loader Monkey Patch (Gemini CLI / Internal Python Tools)

If the agent is a Python-native CLI tool running in local mode (like `Gemini CLI`) that imports and runs MLX in the exact same process, you don't need a web server. You just need to hijack the `mlx_lm.load()` function.

### The Concept
Python is extremely dynamic. We can write a "wrapper" script that launches the agent, but right before the agent boots, we overwrite the standard MLX loading function with our own. When the agent tries to load a model, we secretly strap the TSP `KVCacheManager` onto it.

### The Implementation (`tsp_injector.py`)

```python
import mlx_lm
from tsp_mlx.sparse_cache import KVCacheManager
from tsp_mlx.cortex_hook import CortexHook

# 1. Save the original loader
original_mlx_load = mlx_lm.load

# 2. Define the Trojan Horse loader
def tsp_patched_load(path_or_hf_repo: str, *args, **kwargs):
    print(f"[\u03C4-Gate] Intercepting load request for: {path_or_hf_repo}")
    
    model, tokenizer = original_mlx_load(path_or_hf_repo, *args, **kwargs)
    
    hook = CortexHook(eval_interval=10, threshold=0.1)
    manager = KVCacheManager(hook, model=model, enable_compression=True, enable_consolidation=True)
    model.tsp_kv_manager = manager
    
    print("[\u03C4-Gate] Model successfully patched.")
    return model, tokenizer

# 3. Apply the Monkey Patch globally
mlx_lm.load = tsp_patched_load

# 4. Boot the target agent
import gemini_cli.main # Replace with the actual entry point
gemini_cli.main.run()
```

By patching the loader, the host application continues running normally, but every forward pass is secretly managed by TSP.