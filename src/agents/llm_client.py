import json
import urllib.request
import urllib.error
from src.core.logger import get_logger

import time

logger = get_logger("LLMClient")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_VERSION_URL = "http://localhost:11434/api/version"
OLLAMA_MODEL = "llama3.1"

_ollama_online: bool = None
_last_probe_time: float = 0.0


def is_ollama_online(probe_timeout: float = 0.3) -> bool:
    """Quickly probes if Ollama daemon is running, caching status for 30s to prevent UI lag."""
    global _ollama_online, _last_probe_time
    now = time.time()
    if _ollama_online is not None and (now - _last_probe_time < 30.0):
        return _ollama_online

    try:
        req = urllib.request.Request(OLLAMA_VERSION_URL)
        with urllib.request.urlopen(req, timeout=probe_timeout) as resp:
            _ollama_online = (resp.status == 200)
    except Exception:
        _ollama_online = False
    _last_probe_time = now
    return _ollama_online


def generate_llm_response(prompt: str) -> str:
    """Generates a response from the local Ollama LLM, with instant fallback when daemon is down."""
    if not is_ollama_online():
        raise ConnectionRefusedError("Ollama service offline at localhost:11434")

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(OLLAMA_URL, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get("response", "").strip()
    except Exception as e:
        logger.error(f"Failed to generate LLM response from Ollama: {e}")
        raise
