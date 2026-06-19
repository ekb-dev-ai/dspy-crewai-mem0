# DSPy + Mem0 + Ollama Demo

Production-style reference demo showing how **DSPy** pipelines benefit from **Mem0** long-term memory, with **Ollama** for local LLM inference and a **Streamlit live dashboard** for efficiency metrics.

## 🎥 Demo Video

[![Watch the video](https://img.youtube.com/vi/hJBocmICFgU/maxresdefault.jpg)](https://www.youtube.com/watch?v=hJBocmICFgU)

## What it demonstrates

- **No-memory baseline** — same DSPy pipeline without retrieval; every run starts cold.
- **Memory pipeline** — Mem0 search before planning, then store synthesized reports for reuse.
- **Heavy workload** — ~10-minute multi-task loop with fast/deep routing based on memory hits.
- **Live dashboard** — tracks memory calls, LLM requests, token estimates, and savings across all demo modes.

## Prerequisites

1. **Python 3.10–3.12**
2. **Poetry**
3. **Ollama** running locally at `http://localhost:11434`

Pull the required models:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

## Install

```bash
git clone git@github.com:ekb-dev-ai/dspy-crewai-mem0.git
cd dspy-crewai-mem0
poetry install
```

## Demo commands


| Command                                                         | Description                                         |
| --------------------------------------------------------------- | --------------------------------------------------- |
| `poetry run demo-reset`                                         | Clear Chroma memories, live telemetry, and run logs |
| `poetry run demo-no-memory`                                     | Run the control pipeline (no Mem0 retrieval)        |
| `poetry run demo-memory`                                        | Run the memory-augmented pipeline                   |
| `poetry run demo-heavy --duration-minutes 10 --pause-seconds 4` | Long-running workload with live telemetry           |
| `poetry run demo-dashboard`                                     | Open the Streamlit live dashboard                   |


### Suggested demo flow

**Terminal 1 — dashboard**

```bash
poetry run demo-reset
poetry run demo-dashboard
```

**Terminal 2 — pipelines**

```bash
poetry run demo-no-memory
poetry run demo-memory
poetry run demo-memory   # second run should show memory hits
```

**Terminal 3 — heavy workload (optional)**

```bash
poetry run demo-heavy --duration-minutes 10 --pause-seconds 4
```

The dashboard reads `./data/live_metrics.json` and updates as events are logged.

## Project layout

```
app/
  config.py                 # Mem0 + DSPy/Ollama configuration
  utils.py                  # Run logging and token estimates
  reset_session.py          # Clears session data
  pipelines/
    signatures.py           # GapPlanner, Researcher, Synthesizer
    no_memory_pipeline.py   # Control pipeline
    memory_pipeline.py      # Mem0-augmented pipeline
  telemetry/
    live_telemetry.py       # Writes live_metrics.json
    live_dashboard.py       # Streamlit UI
    run_dashboard.py        # Dashboard entrypoint
  workloads/
    heavy_workload.py       # Multi-task loop with fast/deep routing
data/                       # Gitignored runtime data (Chroma, logs, metrics)
```

## How the memory pipeline works

1. **Search** — Mem0 retrieves relevant memories for the topic (`user_id` + `agent_id` filters).
2. **Plan** — `GapPlanner` identifies open questions not covered by prior memory.
3. **Research** — `Researcher` generates new findings for those gaps.
4. **Synthesize** — `Synthesizer` merges prior memory and new findings without duplication.
5. **Store** — Final report is added back to Mem0 for future runs.

The heavy workload reuses the same Mem0 store and routes tasks to a **fast path** when enough memory is already available, otherwise a **deep path** with full planning and execution.

## Configuration

Defaults in `app/config.py`:


| Setting      | Value                        |
| ------------ | ---------------------------- |
| LLM          | `llama3.1:8b` via Ollama     |
| Embedder     | `nomic-embed-text`           |
| Vector store | Chroma at `./data/chroma_db` |
| Collection   | `dspy_research_memories`     |


## Data files (local only)

These are created at runtime and listed in `.gitignore`:

- `./data/chroma_db/` — Mem0 vector store
- `./data/runs.jsonl` — Per-run JSON logs from pipelines
- `./data/live_metrics.json` — Live dashboard telemetry

Run `poetry run demo-reset` before a clean demo session.

## License

MIT
