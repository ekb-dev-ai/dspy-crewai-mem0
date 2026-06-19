from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

try:
    from app.telemetry.live_telemetry import read_live_metrics
except ModuleNotFoundError:
    # Support execution via `streamlit run app/telemetry/live_dashboard.py`.
    from live_telemetry import read_live_metrics

st.set_page_config(page_title="Mem0 Live Dashboard", layout="wide")
st.title("DSPy + Mem0 Live Dashboard")
st.caption("Tracks live memory and LLM efficiency from demo-no-memory, demo-memory, and demo-heavy.")

refresh_ms = st.sidebar.slider("Refresh interval (ms)", min_value=1000, max_value=10000, value=2000, step=500)
st_autorefresh(interval=refresh_ms, key="mem0_live_refresh")

payload = read_live_metrics()
if not payload:
    st.warning("No telemetry found yet. Run `poetry run demo-no-memory`, `demo-memory`, or `demo-heavy`.")
    st.stop()

totals = payload.get("totals", {})
events = payload.get("events", [])

status = payload.get("status", "unknown")
started_at = datetime.fromtimestamp(payload.get("started_at", 0)).strftime("%Y-%m-%d %H:%M:%S")
last_updated = datetime.fromtimestamp(payload.get("last_updated", 0)).strftime("%Y-%m-%d %H:%M:%S")
st.caption(
    f"Session `{payload.get('session_id', '-')}` | status `{status}` | started `{started_at}` | updated `{last_updated}`"
)

memory_calls = totals.get("memory_search_calls", 0) + totals.get("memory_add_calls", 0)
llm_actual = totals.get("llm_requests", 0)
llm_baseline = totals.get("llm_requests_baseline", 0)
llm_saved = max(0, llm_baseline - llm_actual)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Tasks", totals.get("tasks_processed", 0))
col2.metric("Memory Calls", memory_calls)
col3.metric("Memory Hits", totals.get("memory_hits", 0))
col4.metric("LLM Requests", llm_actual, delta=f"-{llm_saved} vs baseline" if llm_saved else None)
col5.metric("Token Saved (est)", totals.get("token_saved_est", 0))
col6.metric("Cost Saved USD (est)", f"{totals.get('cost_usd_saved_est', 0.0):.4f}")

st.markdown("---")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("No Memory Runs", totals.get("no_memory_count", 0))
col_b.metric("Memory Pipeline Runs", totals.get("memory_pipeline_count", 0))
col_c.metric("Fast Path (heavy)", totals.get("fast_path_count", 0))
col_d.metric("Deep Path (heavy)", totals.get("deep_path_count", 0))

if not events:
    st.info("Telemetry is initialized, waiting for first event.")
    st.stop()

df = pd.DataFrame(events)
df["event_idx"] = range(1, len(df) + 1)
df["cum_memory_hits"] = df["memory_hits"].cumsum()
df["cum_llm_requests"] = df["llm_requests"].cumsum()
df["cum_cost_saved"] = (df["token_est_baseline"] - df["token_est_actual"]).clip(lower=0).cumsum() * 0.004 / 1000.0

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    fig1 = px.line(
        df,
        x="event_idx",
        y=["cum_memory_hits", "cum_llm_requests"],
        title="Cumulative Memory Hits vs LLM Requests",
    )
    st.plotly_chart(fig1, use_container_width=True)

with chart_col2:
    fig2 = px.line(
        df,
        x="event_idx",
        y="cum_cost_saved",
        title="Estimated Cost Saved Over Time (USD)",
    )
    st.plotly_chart(fig2, use_container_width=True)

fig3 = px.histogram(
    df,
    x="path",
    title="Run Type Breakdown (no_memory / memory / fast / deep)",
    color="path",
    barmode="group",
)
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Latest Task Events")
display_cols = [
    "event_idx",
    "iteration",
    "task_id",
    "path",
    "memory_hits",
    "llm_requests",
    "token_est_actual",
    "token_est_baseline",
]
if "mode" in df.columns:
    display_cols.insert(3, "mode")
st.dataframe(df[display_cols].tail(20), use_container_width=True, hide_index=True)
