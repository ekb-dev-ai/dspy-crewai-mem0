from __future__ import annotations

import os

# Quiet CrewAI's anonymous telemetry/OTEL before importing it.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import time
from dataclasses import dataclass
from typing import Any

from app.config import build_crew_llm, memory

FAST_PATH_MIN_HITS = 2

PLANNER_SYSTEM = (
    "You are a Curriculum Planner, a veteran instructional designer. "
    "You turn a course task into a tight, concrete plan. Be brief."
)
WRITER_SYSTEM = (
    "You are a Course Content Writer. You produce clear, non-repetitive course "
    "deliverables that stay consistent with prior decisions."
)
REUSE_SYSTEM = (
    "You are a Course Content Writer with access to prior work. You reuse known "
    "material verbatim where possible and add ONLY what is missing. You never "
    "re-derive what has already been decided. Be concise."
)


@dataclass
class TaskResult:
    task_id: str
    output: str
    tokens: int
    prompt_tokens: int
    completion_tokens: int
    llm_calls: int
    latency_sec: float
    memory_hits: int
    path: str  # "fast", "deep", or "no_memory"


class SharedBrainCrew:
    def __init__(
        self,
        user_id: str,
        agent_id: str,
        top_k: int = 8,
        max_tokens: int = 900,
    ) -> None:
        self.user_id = user_id
        self.agent_id = agent_id
        self.top_k = top_k
        self.llm = build_crew_llm(max_tokens=max_tokens)

    # --- Mem0 shared brain -------------------------------------------------
    def _search_memories(self, query: str) -> list[dict[str, Any]]:
        result = memory.search(
            query,
            filters={"user_id": self.user_id, "agent_id": self.agent_id},
            limit=self.top_k,
        )
        if not result:
            return []
        if isinstance(result, dict):
            return result.get("results", [])
        return result

    @staticmethod
    def _format_memories(memories: list[dict[str, Any]], top: int = 5, max_chars: int = 280) -> str:
        # Bound the recalled context: the fast path only wins if reusing memory is
        # cheaper than writing from scratch, so keep the injected prompt small.
        lines = []
        for item in memories[:top]:
            text = (item.get("memory") or "").strip().replace("\n", " ")
            if text:
                lines.append(f"- {text[:max_chars]}")
        return "\n".join(lines) if lines else "No prior memory."

    def wipe_brain(self) -> int:
        """Delete this crew's memories (the 'amnesia switch'). Returns count removed."""
        try:
            before = self._search_memories(" ")
        except Exception:
            before = []
        try:
            memory.delete_all(user_id=self.user_id, agent_id=self.agent_id)
        except Exception:
            pass
        return len(before)

    def _store(self, task_id: str, output: str, session_id: str, path: str) -> None:
        memory.add(
            output,
            user_id=self.user_id,
            agent_id=self.agent_id,
            metadata={"task_id": task_id, "session_id": session_id, "path": path},
        )

    # --- One measured LLM call (exactly one request) -----------------------
    def _call(self, system: str, user: str) -> tuple[str, tuple[int, int, int]]:
        before = self.llm.get_token_usage_summary()
        text = self.llm.call(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        after = self.llm.get_token_usage_summary()
        delta = (
            int(after.prompt_tokens - before.prompt_tokens),
            int(after.completion_tokens - before.completion_tokens),
            int(after.successful_requests - before.successful_requests),
        )
        return str(text), delta

    # --- Run a single work item -------------------------------------------
    def run_task(
        self,
        task_id: str,
        task_text: str,
        use_memory: bool,
        session_id: str,
        store: bool = True,
    ) -> TaskResult:
        start = time.perf_counter()

        if use_memory:
            memories = self._search_memories(task_text)
            known = self._format_memories(memories)
            hits = len(memories)
        else:
            known = "No prior memory."
            hits = 0

        prompt_tokens = completion_tokens = calls = 0

        def _add(delta: tuple[int, int, int]) -> None:
            nonlocal prompt_tokens, completion_tokens, calls
            prompt_tokens += delta[0]
            completion_tokens += delta[1]
            calls += delta[2]

        if use_memory and hits >= FAST_PATH_MIN_HITS:
            # Fast path: reuse memory in a single call.
            output, delta = self._call(
                REUSE_SYSTEM,
                (
                    f"Task: {task_text}\n\n"
                    f"KNOWN MEMORY (reuse where possible):\n{known}\n\n"
                    "Produce the deliverable, reusing the known memory and adding only the "
                    "missing pieces. Do not re-derive what is already known."
                ),
            )
            _add(delta)
            path = "fast"
        else:
            # Deep path: plan (1 call) then write (1 call).
            plan, plan_delta = self._call(
                PLANNER_SYSTEM,
                (
                    f"Task: {task_text}\n\nKnown context:\n{known}\n\n"
                    "Produce a short, concrete plan (max 6 bullets)."
                ),
            )
            _add(plan_delta)
            output, write_delta = self._call(
                WRITER_SYSTEM,
                f"Plan:\n{plan}\n\nUsing the plan, produce the final deliverable for: {task_text}",
            )
            _add(write_delta)
            path = "deep" if use_memory else "no_memory"

        # Stop the clock before persisting: storage is async/amortized bookkeeping
        # in a real system, so it should not count against "time to answer".
        latency = round(time.perf_counter() - start, 3)

        if use_memory and store:
            self._store(task_id, output, session_id, path)

        return TaskResult(
            task_id=task_id,
            output=output,
            tokens=prompt_tokens + completion_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            llm_calls=calls,
            latency_sec=latency,
            memory_hits=hits,
            path=path,
        )
