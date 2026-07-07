from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import litellm

_LOCK = threading.Lock()
_INSTALLED = False

# Global running totals. Every successful LLM call adds to these regardless of
# whether a measure() block is open, which makes snapshot-deltas race-proof.
_GLOBAL_PROMPT = 0
_GLOBAL_COMPLETION = 0
_GLOBAL_CALLS = 0


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
        }


def _extract_tokens(response_obj: Any) -> tuple[int, int]:
    """Pull (prompt_tokens, completion_tokens) from a LiteLLM response."""
    usage = getattr(response_obj, "usage", None)
    if usage is None and isinstance(response_obj, dict):
        usage = response_obj.get("usage")
    if usage is None:
        return 0, 0

    def _get(obj: Any, key: str) -> int:
        if isinstance(obj, dict):
            value = obj.get(key)
        else:
            value = getattr(obj, key, None)
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    return _get(usage, "prompt_tokens"), _get(usage, "completion_tokens")


def _record(prompt_tokens: int, completion_tokens: int) -> None:
    global _GLOBAL_PROMPT, _GLOBAL_COMPLETION, _GLOBAL_CALLS
    with _LOCK:
        _GLOBAL_PROMPT += prompt_tokens
        _GLOBAL_COMPLETION += completion_tokens
        _GLOBAL_CALLS += 1


def _snapshot() -> tuple[int, int, int]:
    with _LOCK:
        return _GLOBAL_PROMPT, _GLOBAL_COMPLETION, _GLOBAL_CALLS


def _success_callback(kwargs, response_obj, start_time, end_time):  # noqa: ANN001
    prompt_tokens, completion_tokens = _extract_tokens(response_obj)
    if prompt_tokens or completion_tokens:
        _record(prompt_tokens, completion_tokens)


def install_usage_meter() -> None:
    """Register the LiteLLM success callback once. Safe to call repeatedly."""
    global _INSTALLED
    if _INSTALLED:
        return
    # Function-style callbacks reliably carry token usage for the Ollama provider.
    if _success_callback not in litellm.success_callback:
        litellm.success_callback.append(_success_callback)
    if _success_callback not in litellm._async_success_callback:
        litellm._async_success_callback.append(_success_callback)
    _INSTALLED = True


def _flush(settle: float = 0.3, max_wait: float = 4.0) -> None:
    """Wait until background logging stops adding calls (or max_wait elapses)."""
    deadline = time.time() + max_wait
    last = _snapshot()[2]
    stable_since = time.time()
    while time.time() < deadline:
        time.sleep(0.05)
        current = _snapshot()[2]
        if current != last:
            last = current
            stable_since = time.time()
        elif time.time() - stable_since >= settle:
            return


@contextmanager
def measure() -> Iterator[Usage]:
    """Measure real LLM token usage for everything run inside the block.

    The yielded Usage is populated on block exit (after a stabilization flush),
    so read its fields *after* the `with` statement.
    """
    install_usage_meter()
    usage = Usage()
    start_prompt, start_completion, start_calls = _snapshot()
    try:
        yield usage
    finally:
        _flush()
        end_prompt, end_completion, end_calls = _snapshot()
        usage.prompt_tokens = end_prompt - start_prompt
        usage.completion_tokens = end_completion - start_completion
        usage.llm_calls = end_calls - start_calls
