from __future__ import annotations

import json

import dspy

import app.config  # noqa: F401  # Ensure DSPy LM is configured.
from app.pipelines.signatures import GapPlanner, Researcher, Synthesizer
from app.telemetry.live_telemetry import log_pipeline_run
from app.utils import log_run, now_ts, rough_token_count


class NoMemoryPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.plan = dspy.Predict(GapPlanner)
        self.research = dspy.Predict(Researcher)
        self.synthesize = dspy.Predict(Synthesizer)

    def forward(self, topic: str, session_id: str) -> dict:
        start_time = now_ts()
        prior_memory = "No prior memory."
        plan = self.plan(topic=topic, known_facts=prior_memory)
        researched = self.research(topic=topic, open_questions=plan.open_questions)
        synthesis = self.synthesize(
            topic=topic,
            prior_memory=prior_memory,
            new_findings=researched.findings,
        )

        prompt_blob = f"{topic}\n{prior_memory}\n{plan.open_questions}"
        output_blob = f"{researched.findings}\n{synthesis.report}"
        run_record = {
            "mode": "no_memory_pipeline",
            "topic": topic,
            "session_id": session_id,
            "metrics": {
                "latency_sec": round(now_ts() - start_time, 2),
                "memory_hits": 0,
                "recalled_chars": 0,
                "prompt_tokens_est": rough_token_count(prompt_blob),
                "output_tokens_est": rough_token_count(output_blob),
                "novelty_ratio": 0.0,
            },
            "open_questions": plan.open_questions,
            "new_findings": researched.findings,
            "report": synthesis.report,
        }
        log_run(run_record)
        log_pipeline_run(run_record)
        return run_record


def main() -> None:
    pipeline = NoMemoryPipeline()
    topic = "Transformer architecture improvements in 2024-2025"
    print(json.dumps(pipeline(topic=topic, session_id="no_memory_001"), indent=2))


if __name__ == "__main__":
    main()
