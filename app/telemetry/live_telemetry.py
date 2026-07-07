from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

METRICS_PATH = Path("./data/live_metrics.json")
_LOCK = threading.Lock()


def _ensure_data_dir() -> None:
    Path("./data").mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def reset_live_metrics(session_id: str, duration_minutes: int) -> None:
    _ensure_data_dir()
    now = time.time()
    payload = {
        "session_id": session_id,
        "started_at": now,
        "duration_minutes": duration_minutes,
        "last_updated": now,
        "status": "running",
        "totals": {
            "tasks_processed": 0,
            "memory_search_calls": 0,
            "memory_add_calls": 0,
            "memory_hits": 0,
            "llm_requests": 0,
            "llm_requests_baseline": 0,
            "token_est_actual": 0,
            "token_est_baseline": 0,
            "token_saved_est": 0,
            "cost_usd_actual_est": 0.0,
            "cost_usd_baseline_est": 0.0,
            "cost_usd_saved_est": 0.0,
            "fast_path_count": 0,
            "deep_path_count": 0,
            "no_memory_count": 0,
            "memory_pipeline_count": 0,
            "wall_seconds_actual": 0.0,
            "wall_seconds_baseline": 0.0,
            "framework_counts": {},
        },
        "events": [],
        "amnesia_markers": [],
    }
    with _LOCK:
        _atomic_write_json(METRICS_PATH, payload)


def read_live_metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _estimate_cost_usd(tokens_est: int) -> float:
    # Equivalent API spend estimate (for local Ollama this is hypothetical benchmarking value).
    # Assumption: $0.004 per 1K tokens blended input/output.
    return round((tokens_est / 1000.0) * 0.004, 6)


def ensure_live_metrics(session_id: str | None = None) -> None:
    if read_live_metrics():
        return
    reset_live_metrics(
        session_id=session_id or f"demo_{int(time.time())}",
        duration_minutes=0,
    )


def log_pipeline_run(run_record: dict[str, Any]) -> None:
    """Send demo-no-memory / demo-memory runs to the live dashboard."""
    metrics = run_record["metrics"]
    mode = run_record["mode"]
    prompt_tokens = int(metrics["prompt_tokens_est"])
    output_tokens = int(metrics["output_tokens_est"])
    actual_tokens = prompt_tokens + output_tokens
    llm_requests = 3

    if mode == "no_memory_pipeline":
        path = "no_memory"
        memory_search_calls = 0
        memory_add_calls = 0
        memory_hits = 0
        baseline_tokens = actual_tokens
    else:
        path = "memory"
        memory_search_calls = 1
        memory_add_calls = 1
        memory_hits = int(metrics.get("memory_hits", 0))
        baseline_tokens = actual_tokens + max(0, 120 - memory_hits * 15)

    ensure_live_metrics(session_id=run_record.get("session_id"))
    existing = read_live_metrics()
    iteration = len(existing.get("events", [])) + 1 if existing else 1

    log_task_event(
        {
            "session_id": run_record.get("session_id"),
            "iteration": iteration,
            "task_id": mode,
            "task_text": run_record.get("topic", ""),
            "path": path,
            "mode": mode,
            "memory_search_calls": memory_search_calls,
            "memory_add_calls": memory_add_calls,
            "memory_hits": memory_hits,
            "llm_requests": llm_requests,
            "llm_requests_baseline": llm_requests,
            "token_est_actual": actual_tokens,
            "token_est_baseline": baseline_tokens,
            "novelty_ratio": metrics.get("novelty_ratio", 0.0),
        }
    )


def log_task_event(event: dict[str, Any]) -> None:
    with _LOCK:
        payload = read_live_metrics()
        if not payload:
            reset_live_metrics(
                session_id=event.get("session_id", f"demo_{int(time.time())}"),
                duration_minutes=0,
            )
            payload = read_live_metrics()
        if not payload:
            return

        totals = payload["totals"]
        path_label = event.get("path", "deep")
        actual_tokens = int(event.get("token_est_actual", 0))
        baseline_tokens = int(event.get("token_est_baseline", 0))
        saved_tokens = max(0, baseline_tokens - actual_tokens)

        totals["tasks_processed"] += 1
        totals["memory_search_calls"] += int(event.get("memory_search_calls", 0))
        totals["memory_add_calls"] += int(event.get("memory_add_calls", 0))
        totals["memory_hits"] += int(event.get("memory_hits", 0))
        totals["llm_requests"] += int(event.get("llm_requests", 0))
        totals["llm_requests_baseline"] += int(event.get("llm_requests_baseline", 0))
        totals["token_est_actual"] += actual_tokens
        totals["token_est_baseline"] += baseline_tokens
        totals["token_saved_est"] += saved_tokens
        totals["cost_usd_actual_est"] = round(
            totals["cost_usd_actual_est"] + _estimate_cost_usd(actual_tokens), 6
        )
        totals["cost_usd_baseline_est"] = round(
            totals["cost_usd_baseline_est"] + _estimate_cost_usd(baseline_tokens), 6
        )
        totals["cost_usd_saved_est"] = round(
            totals["cost_usd_baseline_est"] - totals["cost_usd_actual_est"], 6
        )

        if path_label == "fast":
            totals["fast_path_count"] += 1
        elif path_label == "no_memory":
            totals["no_memory_count"] += 1
        elif path_label == "memory":
            totals["memory_pipeline_count"] += 1
        else:
            totals["deep_path_count"] += 1

        event["timestamp"] = time.time()
        payload["events"].append(event)
        payload["events"] = payload["events"][-600:]
        payload["last_updated"] = time.time()
        _atomic_write_json(METRICS_PATH, payload)


def log_race_pair(
    session_id: str,
    iteration: int,
    task_id: str,
    task_text: str,
    framework: str,
    memory_side: dict[str, Any],
    nomemory_side: dict[str, Any],
) -> None:
    """Log one head-to-head task: memory variant vs no-memory variant.

    Each *_side dict carries measured values:
        tokens, prompt_tokens, completion_tokens, llm_calls, latency_sec,
        memory_hits (memory side), path (memory side: 'fast'/'deep').

    Totals treat the no-memory variant as the baseline, so savings are the real
    measured delta between the two runs of the same task.
    """
    mem_tokens = int(memory_side.get("tokens", 0))
    nm_tokens = int(nomemory_side.get("tokens", 0))
    mem_calls = int(memory_side.get("llm_calls", 0))
    nm_calls = int(nomemory_side.get("llm_calls", 0))
    mem_latency = float(memory_side.get("latency_sec", 0.0))
    nm_latency = float(nomemory_side.get("latency_sec", 0.0))
    memory_hits = int(memory_side.get("memory_hits", 0))
    path = memory_side.get("path", "deep")

    with _LOCK:
        payload = read_live_metrics()
        if not payload:
            reset_live_metrics(session_id=session_id, duration_minutes=0)
            payload = read_live_metrics()
        if not payload:
            return

        totals = payload["totals"]
        totals["tasks_processed"] += 1
        totals["memory_search_calls"] += 1
        totals["memory_add_calls"] += 1
        totals["memory_hits"] += memory_hits
        totals["llm_requests"] += mem_calls
        totals["llm_requests_baseline"] += nm_calls
        totals["token_est_actual"] += mem_tokens
        totals["token_est_baseline"] += nm_tokens
        totals["token_saved_est"] += max(0, nm_tokens - mem_tokens)
        totals["cost_usd_actual_est"] = round(
            totals["cost_usd_actual_est"] + _estimate_cost_usd(mem_tokens), 6
        )
        totals["cost_usd_baseline_est"] = round(
            totals["cost_usd_baseline_est"] + _estimate_cost_usd(nm_tokens), 6
        )
        totals["cost_usd_saved_est"] = round(
            totals["cost_usd_baseline_est"] - totals["cost_usd_actual_est"], 6
        )
        totals["wall_seconds_actual"] = round(
            totals.get("wall_seconds_actual", 0.0) + mem_latency, 3
        )
        totals["wall_seconds_baseline"] = round(
            totals.get("wall_seconds_baseline", 0.0) + nm_latency, 3
        )
        if path == "fast":
            totals["fast_path_count"] += 1
        else:
            totals["deep_path_count"] += 1
        fw_counts = totals.setdefault("framework_counts", {})
        fw_counts[framework] = fw_counts.get(framework, 0) + 1

        now = time.time()
        base = {
            "session_id": session_id,
            "iteration": iteration,
            "pair_idx": iteration,
            "task_id": task_id,
            "task_text": task_text,
            "framework": framework,
            "timestamp": now,
        }
        payload["events"].append(
            {
                **base,
                "variant": "memory",
                "path": path,
                "memory_hits": memory_hits,
                "llm_requests": mem_calls,
                "token_est_actual": mem_tokens,
                "prompt_tokens": int(memory_side.get("prompt_tokens", 0)),
                "completion_tokens": int(memory_side.get("completion_tokens", 0)),
                "latency_sec": round(mem_latency, 3),
            }
        )
        payload["events"].append(
            {
                **base,
                "variant": "no_memory",
                "path": "no_memory",
                "memory_hits": 0,
                "llm_requests": nm_calls,
                "token_est_actual": nm_tokens,
                "prompt_tokens": int(nomemory_side.get("prompt_tokens", 0)),
                "completion_tokens": int(nomemory_side.get("completion_tokens", 0)),
                "latency_sec": round(nm_latency, 3),
            }
        )
        payload["events"] = payload["events"][-600:]
        payload["last_updated"] = now
        _atomic_write_json(METRICS_PATH, payload)


def log_consistency_probe(
    session_id: str,
    iteration: int,
    framework: str,
    memory_side: dict[str, Any],
    nomemory_side: dict[str, Any],
    scores: dict[str, float],
) -> None:
    """Log the 'it remembered' recall probe as two events with a consistency score.

    This does not touch savings totals — it is a qualitative consistency signal,
    not part of the token race.
    """
    with _LOCK:
        payload = read_live_metrics()
        if not payload:
            reset_live_metrics(session_id=session_id, duration_minutes=0)
            payload = read_live_metrics()
        if not payload:
            return
        now = time.time()
        for variant, side, score_key in (
            ("memory", memory_side, "memory_consistency"),
            ("no_memory", nomemory_side, "cold_consistency"),
        ):
            payload["events"].append(
                {
                    "session_id": session_id,
                    "iteration": iteration,
                    "pair_idx": iteration,
                    "task_id": "recall_probe",
                    "task_text": "Recall the Week 3 theme decided earlier.",
                    "framework": framework,
                    "variant": variant,
                    "path": "recall",
                    "memory_hits": int(side.get("memory_hits", 0)),
                    "llm_requests": int(side.get("llm_calls", 0)),
                    "token_est_actual": int(side.get("tokens", 0)),
                    "latency_sec": round(float(side.get("latency_sec", 0.0)), 3),
                    "consistency": float(scores.get(score_key, 0.0)),
                    "timestamp": now,
                }
            )
        payload["totals"]["recall_consistency_gap"] = float(scores.get("consistency_gap", 0.0))
        payload["events"] = payload["events"][-600:]
        payload["last_updated"] = now
        _atomic_write_json(METRICS_PATH, payload)


def add_amnesia_marker(at_pair_idx: int, deleted: int) -> None:
    """Record that memory was wiped live, anchored to the current task index."""
    with _LOCK:
        payload = read_live_metrics()
        if not payload:
            return
        payload.setdefault("amnesia_markers", []).append(
            {"pair_idx": at_pair_idx, "deleted": deleted, "timestamp": time.time()}
        )
        payload["last_updated"] = time.time()
        _atomic_write_json(METRICS_PATH, payload)


def mark_complete() -> None:
    with _LOCK:
        payload = read_live_metrics()
        if not payload:
            return
        payload["status"] = "completed"
        payload["last_updated"] = time.time()
        _atomic_write_json(METRICS_PATH, payload)
