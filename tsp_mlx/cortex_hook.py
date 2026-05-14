import json
import subprocess
import os
import mlx.core as mx
from typing import Dict, Any, List
import urllib.request
import numpy as np

class CortexHook:
    def __init__(self, daemon_path: str = None, eval_interval: int = 64, threshold: float = 0.015, threat_threshold: float = 999.0):
        if daemon_path is None:
            cache_dir = os.path.expanduser("~/.cache/tsp-mlx")
            os.makedirs(cache_dir, exist_ok=True)
            daemon_path = os.path.join(cache_dir, "tau-gate")
            
        self.daemon_path = daemon_path
        self.eval_interval = eval_interval
        self.threshold = threshold
        self.threat_threshold = threat_threshold
        self.token_counter = 0
        self.edges = set()
        self.last_lambda_2 = 0.0
        
        self._ensure_daemon_exists()
        
        self.daemon = subprocess.Popen(
            [self.daemon_path, "daemon"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )

    def _ensure_daemon_exists(self):
        if not os.path.exists(self.daemon_path):
            print(f"[TSP] Downloading mathematical daemon to {self.daemon_path}...")
            subprocess.run(["curl", "-L", "-o", self.daemon_path, "https://github.com/steph4n-gh/tau-gate/releases/latest/download/tau-gate-darwin-arm64"])
            os.chmod(self.daemon_path, 0o755)

    def evaluate_attention(self, attention_matrix: mx.array, sinks: List[int], position_ids: List[int]) -> Dict[str, Any]:
        self.token_counter += 1
        
        # Collapse heads
        a_2d = mx.mean(attention_matrix, axis=1)[0]
        if len(a_2d.shape) == 2 and a_2d.shape[0] == 1:
            a_sq = mx.squeeze(a_2d, axis=0) # Shape: [S] for decode
        else:
            a_sq = a_2d # Shape: [S, S] for prefill
        
        if len(a_sq.shape) == 2:
            # Prefill phase [L, L]
            a_sym = mx.maximum(a_sq, a_sq.T)
            thresholded = a_sym > self.threshold
            mx.eval(thresholded)
            indices = np.argwhere(np.array(thresholded)).tolist()
            
            for u_rel, v_rel in indices:
                if u_rel != v_rel and u_rel < len(position_ids) and v_rel < len(position_ids):
                    self.edges.add((position_ids[u_rel], position_ids[v_rel]))
        else:
            # Decode phase [L]
            thresholded = a_sq > self.threshold
            mx.eval(thresholded)
            indices = np.argwhere(np.array(thresholded)).tolist()
            
            source_abs = position_ids[-1]
            for target_rel in indices:
                if target_rel[0] >= len(position_ids): continue
                target_abs = position_ids[target_rel[0]]
                if source_abs != target_abs:
                    self.edges.add((source_abs, target_abs))
                    self.edges.add((target_abs, source_abs))
                    
        if self.token_counter % self.eval_interval != 0:
            return {"action": "ALLOW", "island_indices": []}

        payload = {
            "edges": list(self.edges),
            "sinks": sinks,
            "threat_threshold": self.threat_threshold
        }

        self.daemon.stdin.write(json.dumps(payload) + "\n")
        self.daemon.stdin.flush()

        response_line = self.daemon.stdout.readline()
        if not response_line:
            raise RuntimeError("CortexHook: Daemon connection lost.")
            
        decision = json.loads(response_line)
        self.last_lambda_2 = decision.get("connectivity_score", 0.0)
        return decision

    def __del__(self):
        if hasattr(self, "daemon") and self.daemon.poll() is None:
            self.daemon.terminate()
