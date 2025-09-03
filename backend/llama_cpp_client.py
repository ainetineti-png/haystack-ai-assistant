import subprocess
import json
import os
from typing import Optional

class LlamaCppClient:
    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = 4):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate text using llama.cpp"""
        try:
            cmd = [
                "llama-cli",  # or "main" depending on your llama.cpp build
                "-m", self.model_path,
                "-p", prompt,
                "-n", str(max_tokens),
                "--temp", str(temperature),
                "-c", str(self.n_ctx),
                "-t", str(self.n_threads),
                "--no-display-prompt",
                "--silent-prompt"
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=60  # 1 minute timeout
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"[ERROR] llama.cpp failed: {result.stderr}")
                return "Error: Failed to generate response"
                
        except subprocess.TimeoutExpired:
            return "Error: Response generation timed out"
        except Exception as e:
            print(f"[ERROR] llama.cpp error: {e}")
            return f"Error: {str(e)}"

# Fallback to Ollama if llama.cpp fails
def generate_with_fallback(prompt: str, llama_client: Optional[LlamaCppClient] = None) -> str:
    if llama_client:
        response = llama_client.generate(prompt)
        if not response.startswith("Error"):
            return response
    
    # Fallback to Ollama
    import requests
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3:latest",  # Use available model
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7
            },
            timeout=30
        )
        response.raise_for_status()
        result = response.json().get("response", "")
        if not result.strip():
            return "Error: Empty response from Ollama"
        return result
    except Exception as e:
        print(f"[ERROR] Ollama fallback failed: {e}")
        return f"Error: Both llama.cpp and Ollama failed - {str(e)}"
