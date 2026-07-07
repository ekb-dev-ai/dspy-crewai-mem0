# DSPy + CrewAI + Mem0 + Ollama Demo

Production-style reference demo showing how **DSPy** and **CrewAI** pipelines benefit from **Mem0** long-term memory, with **Ollama** for local LLM inference and a **Streamlit live dashboard** for *measured* efficiency metrics.

## 🎥 Demo Video

### Part1:
[![Watch the video](https://img.youtube.com/vi/hJBocmICFgU/maxresdefault.jpg)](https://www.youtube.com/watch?v=hJBocmICFgU)

### Part2:
[![Watch the video](https://img.youtube.com/vi/-AwX_ErkeKc/maxresdefault.jpg)](https://www.youtube.com/watch?v=-AwX_ErkeKc)

## What it demonstrates

- **No-memory baseline** — same DSPy pipeline without retrieval; every run starts cold.
- **Memory pipeline** — Mem0 search before planning, then store synthesized reports for reuse.
- **Heavy workload** — ~10-minute multi-task loop with fast/deep routing based on memory hits.
- **CrewAI shared-brain race** — runs the *same* tasks through a memory-backed crew and a cold crew, head to head, on the same Mem0 brain. As the brain fills, the memory crew increasingly takes the fast path and its cumulative tokens / cost / latency visibly diverge from the cold crew.
- **"It remembered" probe** — a recall task that depends on a decision made many tasks earlier; the memory crew stays consistent (high embedding similarity to the real earlier decision), the cold crew drifts.
- **Live dashboard** — animated odometers (tokens / cost / seconds / calls saved), a memory-vs-cold race chart, a pulsing shared-brain graph, and a framework filter.

### Measured, not estimated

Token counts are real. CrewAI numbers come from CrewAI's own `get_token_usage_summary()` (one LLM call per step, deterministic); any LiteLLM-routed work is captured by `app/telemetry/usage_meter.py`. The "cost saved" figure is a projection that multiplies *measured* saved tokens by a published frontier-API blended rate (`$0.004 / 1K`, shown on the dashboard).

## Prerequisites

1. **Python 3.10–3.12**
2. **Poetry**
3. **Ollama** running locally at `http://localhost:11434`

Pull the required models:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

The chat model is resolved at startup from what the Ollama server actually has
loaded (preference order `llama3.1:8b → llama3.1:latest → llama3.2:latest → gemma3:27b`),
so the demo keeps working if tags differ. Override with `DEMO_CHAT_MODEL` /
`DEMO_EMBED_MODEL` / `OLLAMA_BASE_URL` env vars.

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
| `poetry run demo-heavy --duration-minutes 10 --pause-seconds 4` | Long-running DSPy workload with live telemetry      |
| `poetry run demo-crew-race --rounds 2 --pause-seconds 1`        | CrewAI memory-vs-cold head-to-head race (the viral one) |
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

### Recording flow (the CrewAI race)

```bash
# Terminal 1
poetry run demo-reset
poetry run demo-dashboard

# Terminal 2 — runs the same 8 course-building tasks twice, memory crew vs cold crew
poetry run demo-crew-race --rounds 2 --pause-seconds 1
```

Watch the dashboard: round 1 fills the shared brain (both crews cost about the
same), and in round 2 the memory crew flips to the fast path and the cost /
token / latency lines fan apart. The run ends with the **"it remembered"** recall
probe — the memory crew names the exact Week 3 theme it defined 15 tasks earlier;
the cold crew guesses. The consistency gap is shown as two embedding-similarity
scores.

#### How CrewAI uses the shared brain

Each task is driven through CrewAI's own `LLM` with explicit role prompts
(Curriculum Planner → Course Content Writer). We call `LLM.call` directly instead
of the multi-agent `Crew` executor: on a local llama model the executor re-prompts
unpredictably (a single task ballooned to 7 LLM calls), which makes token counts
noisy. `LLM.call` is exactly one request per step, so the demo is deterministic
(**deep = 2 calls, fast = 1 call**) and every token is measured.

- **Memory crew** searches Mem0 for prior deliverables. ≥2 hits → fast path
  (reuse + deltas, 1 call); otherwise deep path (plan + write, 2 calls). It writes
  each deliverable back to the shared brain.
- **Cold crew** never touches Mem0 and always runs the deep path.

The brain is the *same* Mem0/Chroma store the DSPy demo uses, so a CrewAI crew can
benefit from memories a DSPy pipeline wrote — two frameworks, one brain.

#### The amnesia switch

While a race is running, the dashboard sidebar has a **💥 Wipe memory now** button.
Press it and the memory crew's brain is deleted mid-run — its next tasks miss
every lookup, fall back to the deep path, and its cost / tokens / latency snap
back up to the cold crew's level. A dashed `🧠 wiped` marker is dropped on the
race charts at that point, and you can watch it slowly re-learn afterwards.

Mechanics: the dashboard only writes a small request file
(`app/telemetry/control.py`); the running race process — which owns the Mem0
handle — performs the wipe between tasks, so the two processes never write Chroma
at once.

## Project layout

```
app/
  config.py                 # Mem0 + DSPy/CrewAI/Ollama config, runtime model resolution
  utils.py                  # Run logging and token estimates
  reset_session.py          # Clears session data
  pipelines/
    signatures.py           # GapPlanner, Researcher, Synthesizer
    no_memory_pipeline.py   # DSPy control pipeline
    memory_pipeline.py      # DSPy Mem0-augmented pipeline
  crew/
    tasks.py                # Course-building work items + recall probe
    shared_brain.py         # CrewAI pipeline on the shared Mem0 brain (fast/deep)
    consistency.py          # Embedding-similarity "it remembered" scoring
  telemetry/
    usage_meter.py          # Measured LiteLLM token usage (context manager)
    live_telemetry.py       # Writes live_metrics.json (race pairs, recall probe)
    live_dashboard.py       # Streamlit UI (odometers, race chart, brain graph)
    run_dashboard.py        # Dashboard entrypoint
  workloads/
    heavy_workload.py       # DSPy multi-task loop with fast/deep routing
    crew_race.py            # CrewAI memory-vs-cold head-to-head race
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
