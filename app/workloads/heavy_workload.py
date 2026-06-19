from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

import dspy

from app.config import memory
from app.telemetry.live_telemetry import log_task_event, mark_complete, reset_live_metrics
from app.utils import rough_token_count


class PlanTask(dspy.Signature):
    """Generate an execution plan for a training task."""

    task = dspy.InputField()
    known_memory = dspy.InputField()
    plan = dspy.OutputField(desc="Step-by-step approach and key talking points.")


class ExecuteTask(dspy.Signature):
    """Execute one task with concrete output."""

    task = dspy.InputField()
    plan = dspy.InputField()
    output = dspy.OutputField(desc="Structured deliverable with concise reasoning.")


class QuickTask(dspy.Signature):
    """Fast-path response when memory already covers the task."""

    task = dspy.InputField()
    known_memory = dspy.InputField()
    output = dspy.OutputField(desc="Updated answer using known memory and only deltas.")


@dataclass
class WorkItem:
    task_id: str
    task_text: str


WORK_ITEMS = [
    WorkItem("module_outline", "Design a complete module outline for lesson 1 on transformer attention."),
    WorkItem("quiz_bank", "Generate 10 assessment questions and answer key for lesson 1."),
    WorkItem("misconceptions", "List top misconceptions students have about attention and how to correct them."),
    WorkItem("lab_guide", "Create lab instructions for implementing a toy attention block."),
    WorkItem("grading_rubric", "Draft grading rubric for the attention lab with 4 proficiency levels."),
    WorkItem("faq", "Create FAQ responses for common student support tickets about transformers."),
    WorkItem("capstone", "Define a capstone project brief using transformer improvements from 2024-2025."),
    WorkItem("stakeholder_brief", "Write a short stakeholder update on course progress and learning outcomes."),
]


class HeavyUsageRunner:
    def __init__(self, user_id: str, agent_id: str, top_k: int = 8):
        self.user_id = user_id
        self.agent_id = agent_id
        self.top_k = top_k
        self.plan = dspy.Predict(PlanTask)
        self.execute = dspy.Predict(ExecuteTask)
        self.quick = dspy.Predict(QuickTask)

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
    def _format_memories(memories: list[dict[str, Any]]) -> str:
        if not memories:
            return "No prior memory."
        lines = []
        for item in memories:
            text = item.get("memory", "").strip()
            if text:
                lines.append(f"- {text}")
        return "\n".join(lines) if lines else "No prior memory."

    def run_task(self, work_item: WorkItem, session_id: str, iteration: int) -> dict[str, Any]:
        memories = self._search_memories(work_item.task_text)
        known_memory = self._format_memories(memories)
        memory_hits = len(memories)

        # Baseline assumes deep path every time.
        llm_requests_baseline = 2

        if memory_hits >= 2:
            # Fast path: one request because memory already covers most of the task.
            quick = self.quick(task=work_item.task_text, known_memory=known_memory)
            final_output = quick.output
            llm_requests_actual = 1
            path = "fast"
        else:
            # Deep path: plan + execute.
            planned = self.plan(task=work_item.task_text, known_memory=known_memory)
            executed = self.execute(task=work_item.task_text, plan=planned.plan)
            final_output = executed.output
            llm_requests_actual = 2
            path = "deep"

        memory.add(
            final_output,
            user_id=self.user_id,
            agent_id=self.agent_id,
            metadata={
                "session_id": session_id,
                "task_id": work_item.task_id,
                "iteration": iteration,
                "path": path,
            },
        )

        actual_blob = f"{work_item.task_text}\n{known_memory}\n{final_output}"
        baseline_blob = f"{work_item.task_text}\nNo prior memory.\n{final_output}"
        actual_tokens = rough_token_count(actual_blob)
        baseline_tokens = rough_token_count(baseline_blob) + 120

        event = {
            "session_id": session_id,
            "iteration": iteration,
            "task_id": work_item.task_id,
            "task_text": work_item.task_text,
            "path": path,
            "memory_search_calls": 1,
            "memory_add_calls": 1,
            "memory_hits": memory_hits,
            "llm_requests": llm_requests_actual,
            "llm_requests_baseline": llm_requests_baseline,
            "token_est_actual": actual_tokens,
            "token_est_baseline": baseline_tokens,
        }
        log_task_event(event)
        return event


def run_live_session(duration_minutes: int, pause_seconds: int) -> None:
    session_id = f"live_{uuid.uuid4().hex[:8]}"
    reset_live_metrics(session_id=session_id, duration_minutes=duration_minutes)
    runner = HeavyUsageRunner(user_id="team_alpha", agent_id="dspy_course_builder", top_k=8)

    started_at = time.time()
    max_seconds = duration_minutes * 60
    iteration = 0

    while (time.time() - started_at) < max_seconds:
        work_item = WORK_ITEMS[iteration % len(WORK_ITEMS)]
        event = runner.run_task(work_item=work_item, session_id=session_id, iteration=iteration + 1)
        print(json.dumps(event, ensure_ascii=True))
        iteration += 1
        time.sleep(max(0, pause_seconds))

    mark_complete()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run heavy DSPy+Mem0 workload with live telemetry.")
    parser.add_argument("--duration-minutes", type=int, default=10)
    parser.add_argument("--pause-seconds", type=int, default=4)
    args = parser.parse_args()
    run_live_session(duration_minutes=args.duration_minutes, pause_seconds=args.pause_seconds)


if __name__ == "__main__":
    main()
