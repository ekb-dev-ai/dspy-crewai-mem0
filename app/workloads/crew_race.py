from __future__ import annotations

import argparse
import json
import time
import uuid

from app.crew.consistency import score_recall
from app.crew.shared_brain import SharedBrainCrew
from app.crew.tasks import RECALL_PROBE, WORK_ITEMS
from app.telemetry.control import clear_amnesia_request, pop_amnesia_request
from app.telemetry.live_telemetry import (
    add_amnesia_marker,
    log_consistency_probe,
    log_race_pair,
    mark_complete,
    reset_live_metrics,
)

USER_ID = "team_alpha"
AGENT_ID = "crew_course_builder"


def _side(result) -> dict:
    return {
        "tokens": result.tokens,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "llm_calls": result.llm_calls,
        "latency_sec": result.latency_sec,
        "memory_hits": result.memory_hits,
        "path": result.path,
    }


def run_race(rounds: int, pause_seconds: int, max_tokens: int) -> None:
    session_id = f"crew_{uuid.uuid4().hex[:8]}"
    reset_live_metrics(session_id=session_id, duration_minutes=0)
    crew = SharedBrainCrew(
        user_id=USER_ID, agent_id=AGENT_ID, top_k=8, max_tokens=max_tokens
    )

    memory_outputs: dict[str, str] = {}
    last_amnesia_id: str | None = None
    iteration = 0
    for round_idx in range(rounds):
        for item in WORK_ITEMS:
            iteration += 1

            # Amnesia switch: if the dashboard requested a wipe, do it here (this
            # process owns the Mem0 handle) and drop a marker on the race chart.
            new_request = pop_amnesia_request(last_amnesia_id)
            if new_request:
                deleted = crew.wipe_brain()
                add_amnesia_marker(at_pair_idx=iteration - 1, deleted=deleted)
                last_amnesia_id = new_request
                clear_amnesia_request()
                print(json.dumps({"amnesia": True, "deleted": deleted, "at": iteration - 1}))

            # Cold crew first (never reads/writes the brain), then memory crew.
            cold = crew.run_task(
                item.task_id, item.task_text, use_memory=False, session_id=session_id, store=False
            )
            mem = crew.run_task(
                item.task_id, item.task_text, use_memory=True, session_id=session_id, store=True
            )
            memory_outputs[item.task_id] = mem.output

            log_race_pair(
                session_id=session_id,
                iteration=iteration,
                task_id=item.task_id,
                task_text=item.task_text,
                framework="crewai",
                memory_side=_side(mem),
                nomemory_side=_side(cold),
            )

            saved = max(0, cold.tokens - mem.tokens)
            print(
                json.dumps(
                    {
                        "round": round_idx + 1,
                        "iteration": iteration,
                        "task_id": item.task_id,
                        "mem_path": mem.path,
                        "mem_hits": mem.memory_hits,
                        "mem_tokens": mem.tokens,
                        "cold_tokens": cold.tokens,
                        "tokens_saved": saved,
                        "mem_latency": mem.latency_sec,
                        "cold_latency": cold.latency_sec,
                    },
                    ensure_ascii=True,
                )
            )
            time.sleep(max(0, pause_seconds))

    # --- The 'it remembered' moment: recall a decision made many tasks ago ---
    reference = memory_outputs.get("syllabus", "")
    if reference:
        iteration += 1
        cold = crew.run_task(
            RECALL_PROBE.task_id, RECALL_PROBE.task_text,
            use_memory=False, session_id=session_id, store=False,
        )
        mem = crew.run_task(
            RECALL_PROBE.task_id, RECALL_PROBE.task_text,
            use_memory=True, session_id=session_id, store=False,
        )
        scores = score_recall(mem.output, cold.output, reference)
        log_consistency_probe(
            session_id=session_id, iteration=iteration, framework="crewai",
            memory_side=_side(mem), nomemory_side=_side(cold), scores=scores,
        )
        print(json.dumps({"recall_probe": scores}, ensure_ascii=True))

    mark_complete()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CrewAI memory-vs-cold race.")
    parser.add_argument("--rounds", type=int, default=2, help="Passes over the task list.")
    parser.add_argument("--pause-seconds", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=900)
    args = parser.parse_args()
    run_race(
        rounds=args.rounds,
        pause_seconds=args.pause_seconds,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
