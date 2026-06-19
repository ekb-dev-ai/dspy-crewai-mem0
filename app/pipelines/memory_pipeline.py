from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import dspy

from app.config import memory
from app.pipelines.signatures import GapPlanner, Researcher, Synthesizer
from app.telemetry.live_telemetry import log_pipeline_run
from app.utils import log_run, now_ts, rough_token_count


@dataclass
class RunMetrics:
    latency_sec: float
    memory_hits: int
    recalled_chars: int
    prompt_tokens_est: int
    output_tokens_est: int
    novelty_ratio: float


class MemoryPipeline(dspy.Module):
    def __init__(self, user_id: str, agent_id: str, top_k: int = 8):
        super().__init__()
        self.user_id = user_id
        self.agent_id = agent_id
        self.top_k = top_k
        self.plan = dspy.Predict(GapPlanner)
        self.research = dspy.Predict(Researcher)
        self.synthesize = dspy.Predict(Synthesizer)

    def _search_memories(self, topic: str) -> list[dict[str, Any]]:
        result = memory.search(
            topic,
            filters={"user_id": self.user_id, "agent_id": self.agent_id},
            limit=self.top_k,
        )
        if not result:
            return []
        if isinstance(result, dict):
            return result.get("results", [])
        return result

    @staticmethod
    def _format_memories(memories: list[dict[str, Any]]) -> str:
        if not memories:
            return "No prior memory."
        lines = []
        for memory_item in memories:
            memory_text = memory_item.get("memory", "").strip()
            if memory_text:
                lines.append(f"- {memory_text}")
        return "\n".join(lines) if lines else "No prior memory."

    @staticmethod
    def _novelty_ratio(prior: str, new: str) -> float:
        prior_set = {line.strip().lower() for line in prior.splitlines() if line.strip()}
        new_set = {line.strip().lower() for line in new.splitlines() if line.strip()}
        if not new_set:
            return 0.0
        overlap = len(prior_set.intersection(new_set))
        return round((len(new_set) - overlap) / len(new_set), 3)

    def forward(self, topic: str, session_id: str) -> dict[str, Any]:
        start_time = now_ts()
        memories = self._search_memories(topic)
        prior_memory = self._format_memories(memories)

        plan = self.plan(topic=topic, known_facts=prior_memory)
        researched = self.research(topic=topic, open_questions=plan.open_questions)
        synthesis = self.synthesize(
            topic=topic,
            prior_memory=prior_memory,
            new_findings=researched.findings,
        )

        memory.add(
            synthesis.report,
            user_id=self.user_id,
            agent_id=self.agent_id,
            metadata={"topic": topic, "session_id": session_id, "stage": "final_report"},
        )

        prompt_blob = f"{topic}\n{prior_memory}\n{plan.open_questions}"
        output_blob = f"{researched.findings}\n{synthesis.report}"
        novelty = self._novelty_ratio(prior_memory, researched.findings)
        metrics = RunMetrics(
            latency_sec=round(now_ts() - start_time, 2),
            memory_hits=len(memories),
            recalled_chars=len(prior_memory),
            prompt_tokens_est=rough_token_count(prompt_blob),
            output_tokens_est=rough_token_count(output_blob),
            novelty_ratio=novelty,
        )

        run_record = {
            "mode": "memory_pipeline",
            "topic": topic,
            "session_id": session_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "metrics": metrics.__dict__,
            "open_questions": plan.open_questions,
            "new_findings": researched.findings,
            "report": synthesis.report,
        }
        log_run(run_record)
        log_pipeline_run(run_record)
        return run_record


def main() -> None:
    pipeline = MemoryPipeline(
        user_id="team_alpha",
        agent_id="dspy_researcher",
        top_k=8,
    )
    topic = "Transformer architecture improvements in 2024-2025"
    print(json.dumps(pipeline(topic=topic, session_id="run_001"), indent=2))


if __name__ == "__main__":
    main()
