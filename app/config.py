import io
import os
import warnings
from contextlib import redirect_stderr

# Keep the recording terminal clean. DSPy resets the warnings filters during its
# own import and emits a wall of DeprecationWarnings from its avatar signatures,
# so a normal filter can't catch them. Redirect stderr for just the import (real
# import failures still raise exceptions, so nothing important is hidden).
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="SelectableGroups dict interface is deprecated. Use select.",
)
with redirect_stderr(io.StringIO()):
    from mem0 import Memory
    import dspy

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# The target Ollama server hosts a changing set of model tags, so resolve the chat
# model at startup from whatever is actually loaded (env override wins).
_CHAT_PREFERENCES = ["llama3.1:8b", "llama3.1:latest", "llama3.2:latest", "gemma3:27b"]
_FALLBACK_CHAT_MODEL = "llama3.1:8b"


def _list_ollama_models() -> list[str]:
    try:
        import urllib.request

        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5) as resp:
            import json as _json

            data = _json.loads(resp.read().decode("utf-8"))
        return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        return []


def resolve_chat_model() -> str:
    override = os.environ.get("DEMO_CHAT_MODEL")
    if override:
        return override
    available = _list_ollama_models()
    for preferred in _CHAT_PREFERENCES:
        if preferred in available:
            return preferred
    # Fall back to any non-embedding chat model that is loaded, else a sane default.
    for name in available:
        if "embed" not in name and ":cloud" not in name:
            return name
    return _FALLBACK_CHAT_MODEL


CHAT_MODEL = resolve_chat_model()
EMBED_MODEL = os.environ.get("DEMO_EMBED_MODEL", "nomic-embed-text")
MEMORY_COLLECTION = "dspy_research_memories"

MEM0_CONFIG = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": CHAT_MODEL,
            "temperature": 0,
            "max_tokens": 1200,
            "ollama_base_url": OLLAMA_BASE_URL,
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": EMBED_MODEL,
            "ollama_base_url": OLLAMA_BASE_URL,
        },
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": MEMORY_COLLECTION,
            "path": "./data/chroma_db",
        },
    },
}

# Shared Mem0 brain. Both the DSPy and CrewAI pipelines read/write this same store.
memory = Memory.from_config(MEM0_CONFIG)

# DSPy with Ollama through LiteLLM routing.
lm = dspy.LM(
    model=f"ollama_chat/{CHAT_MODEL}",
    api_base=OLLAMA_BASE_URL,
    temperature=0.2,
    max_tokens=1200,
)
dspy.settings.configure(lm=lm)


def build_crew_llm(temperature: float = 0.2, max_tokens: int = 1200):
    """CrewAI LLM bound to the same Ollama server (routes through LiteLLM)."""
    from crewai import LLM

    return LLM(
        model=f"ollama/{CHAT_MODEL}",
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )
