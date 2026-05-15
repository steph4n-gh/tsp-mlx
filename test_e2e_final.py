import requests
import json
import time

def run_test():
    print("Starting End-to-End Verification Test...")
    url = "http://127.0.0.1:8080/v1/chat/completions"
    payload = {
        "model": "tsp-qwen-7b",
        "messages": [
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": "Explain the history of the Roman Empire in great detail, covering at least 1000 words. " * 5}
        ],
        "stream": True,
        "max_tokens": 500
    }
    
    start_time = time.time()
    response = requests.post(url, json=payload, stream=True)
    
    tokens = 0
    final_stats = {}
    
    for line in response.iter_lines():
        if line:
            decoded = line.decode('utf-8')
            if decoded.startswith("data: "):
                data_str = decoded[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0].get("delta", {}).get("content", "")
                        if content:
                            tokens += 1
                    if "tsp_stats" in data:
                        final_stats = data["tsp_stats"]
                except Exception as e:
                    print(f"Error parsing JSON: {e}")
                    
    elapsed = time.time() - start_time
    tps = tokens / elapsed if elapsed > 0 else 0
    
    print("\n--- TEST RESULTS ---")
    print(f"Generation complete: {tokens} tokens in {elapsed:.2f} seconds ({tps:.2f} tokens/sec).")
    print(f"Final Stats from backend: {final_stats}")
    
    if tokens > 10 and final_stats:
        print("\n\033[1;32m[SUCCESS] E2E Test Passed: Stream is stable, stats are populated, no OOM occurred.\033[0m")
    else:
        print("\n\033[1;31m[FAILURE] E2E Test Failed: Stream collapsed or stats missing.\033[0m")

if __name__ == "__main__":
    run_test()
