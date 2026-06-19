import warnings

from mem0 import Memory
import dspy

# Keep runtime output clean by suppressing a known third-party deprecation warning.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="SelectableGroups dict interface is deprecated. Use select.",
)

OLLAMA_BASE_URL = "http://localhost:11434"

MEM0_CONFIG = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.1:8b",
            "temperature": 0,
            "max_tokens": 1200,
            "ollama_base_url": OLLAMA_BASE_URL,
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
            "ollama_base_url": OLLAMA_BASE_URL,
        },
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "dspy_research_memories",
            "path": "./data/chroma_db",
        },
    },
}

memory = Memory.from_config(MEM0_CONFIG)

# DSPy with Ollama through LiteLLM routing.
lm = dspy.LM(
    model="ollama_chat/llama3.1:8b",
    api_base=OLLAMA_BASE_URL,
    temperature=0.2,
    max_tokens=1200,
)
dspy.settings.configure(lm=lm)
