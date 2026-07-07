from __future__ import annotations

import json
import math
import urllib.request

from app.config import EMBED_MODEL, OLLAMA_BASE_URL


def embed(text: str) -> list[float]:
    payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("embedding", [])


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def consistency_score(answer: str, reference: str) -> float:
    """Cosine similarity (0..1, higher = more consistent with the reference)."""
    return round(max(0.0, cosine(embed(answer), embed(reference))), 4)


def score_recall(memory_answer: str, cold_answer: str, reference: str) -> dict[str, float]:
    """Compare a memory recall vs a cold answer against the real earlier decision."""
    ref_vec = embed(reference)
    mem = round(max(0.0, cosine(embed(memory_answer), ref_vec)), 4)
    cold = round(max(0.0, cosine(embed(cold_answer), ref_vec)), 4)
    return {
        "memory_consistency": mem,
        "cold_consistency": cold,
        "consistency_gap": round(mem - cold, 4),
    }
