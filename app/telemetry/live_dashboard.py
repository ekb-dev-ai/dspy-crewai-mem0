from __future__ import annotations

import math
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

try:
    from app.telemetry.live_telemetry import read_live_metrics
    from app.telemetry.control import request_amnesia
except ModuleNotFoundError:
    # Support execution via `streamlit run app/telemetry/live_dashboard.py`.
    from live_telemetry import read_live_metrics
    from control import request_amnesia

COST_PER_1K = 0.004  # blended frontier-API $/1K tokens used for the projection
MEM_COLOR = "#22d3ee"
COLD_COLOR = "#fb7185"
GOOD_COLOR = "#34d399"

st.set_page_config(page_title="Mem0 Live Dashboard", layout="wide")

# --- light global styling for a clean recording look -----------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem;}
      #MainMenu, footer {visibility: hidden;}
      h1, h2, h3 {letter-spacing: .5px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 Mem0 Shared-Brain Live Dashboard")
st.caption("Measured CrewAI + DSPy efficiency on a shared Mem0 brain — memory vs cold, head to head.")

refresh_ms = st.sidebar.slider("Refresh interval (ms)", 1000, 10000, 2000, 500)
st_autorefresh(interval=refresh_ms, key="mem0_live_refresh")

# --- the amnesia switch ----------------------------------------------------
st.sidebar.markdown("### 🧠💥 Amnesia switch")
st.sidebar.caption("Wipe the memory crew's brain mid-run and watch its cost snap back to cold.")
if st.sidebar.button("💥 Wipe memory now", use_container_width=True):
    rid = request_amnesia()
    st.sidebar.warning(f"Wipe requested ({rid}). It triggers before the next task.")

payload = read_live_metrics()
if not payload:
    st.warning("No telemetry yet. Run `poetry run demo-crew-race` (or `demo-heavy` / `demo-memory`).")
    st.stop()

totals = payload.get("totals", {})
events = payload.get("events", [])
amnesia_markers = payload.get("amnesia_markers", [])
df = pd.DataFrame(events) if events else pd.DataFrame()

# --- framework filter ------------------------------------------------------
frameworks = sorted(df["framework"].dropna().unique().tolist()) if "framework" in df else []
if frameworks:
    options = ["all"] + frameworks
    chosen = st.sidebar.selectbox("Framework", options, index=0)
    if chosen != "all":
        df = df[df["framework"] == chosen]

status = payload.get("status", "unknown")
started_at = datetime.fromtimestamp(payload.get("started_at", 0)).strftime("%H:%M:%S")
last_updated = datetime.fromtimestamp(payload.get("last_updated", 0)).strftime("%H:%M:%S")
fw_counts = totals.get("framework_counts", {})
fw_label = " · ".join(f"{k}:{v}" for k, v in fw_counts.items()) or "—"
st.caption(
    f"session `{payload.get('session_id','-')}` · status `{status}` · "
    f"started `{started_at}` · updated `{last_updated}` · frameworks `{fw_label}`"
)


# --- animated odometer -----------------------------------------------------
def odometer(col, label, value, key, *, prefix="", decimals=0, color=MEM_COLOR):
    prev = st.session_state.get(key, 0.0)
    st.session_state[key] = float(value)
    uid = key.replace(" ", "_")
    with col:
        components.html(
            f"""
            <div style="font-family:ui-monospace,Menlo,monospace;text-align:center;
                        background:linear-gradient(180deg,#141b32,#0b1020);
                        border:1px solid #243056;border-radius:14px;padding:14px 8px;">
              <div style="color:#9fb0d9;font-size:12px;text-transform:uppercase;
                          letter-spacing:1px;">{label}</div>
              <div id="{uid}" style="color:{color};font-size:34px;font-weight:700;
                          text-shadow:0 0 18px {color}55;margin-top:6px;">{prefix}0</div>
            </div>
            <script>
              (function(){{
                const el=document.getElementById("{uid}");
                const from={float(prev)}, to={float(value)}, dur=900, t0=performance.now();
                function fmt(n){{return "{prefix}"+n.toLocaleString(undefined,
                    {{minimumFractionDigits:{decimals},maximumFractionDigits:{decimals}}});}}
                function tick(t){{
                  let p=Math.min(1,(t-t0)/dur); p=1-Math.pow(1-p,3);
                  el.textContent=fmt(from+(to-from)*p);
                  if(p<1) requestAnimationFrame(tick);
                }}
                requestAnimationFrame(tick);
              }})();
            </script>
            """,
            height=110,
        )


# --- headline numbers (prefer measured race pairs when present) ------------
if not df.empty and "variant" in df.columns:
    mem_df = df[df["variant"] == "memory"]
    cold_df = df[df["variant"] == "no_memory"]
    mem_tokens = int(mem_df["token_est_actual"].sum())
    cold_tokens = int(cold_df["token_est_actual"].sum())
    tokens_saved = max(0, cold_tokens - mem_tokens)
    mem_latency = float(mem_df.get("latency_sec", pd.Series(dtype=float)).sum())
    cold_latency = float(cold_df.get("latency_sec", pd.Series(dtype=float)).sum())
    mem_calls = int(mem_df["llm_requests"].sum())
    cold_calls = int(cold_df["llm_requests"].sum())
    tasks = int(len(mem_df))
    memory_hits = int(mem_df["memory_hits"].sum()) if "memory_hits" in mem_df else 0
else:
    mem_tokens = int(totals.get("token_est_actual", 0))
    cold_tokens = int(totals.get("token_est_baseline", 0))
    tokens_saved = int(totals.get("token_saved_est", 0))
    mem_latency = float(totals.get("wall_seconds_actual", 0.0))
    cold_latency = float(totals.get("wall_seconds_baseline", 0.0))
    mem_calls = int(totals.get("llm_requests", 0))
    cold_calls = int(totals.get("llm_requests_baseline", 0))
    tasks = int(totals.get("tasks_processed", 0))
    memory_hits = int(totals.get("memory_hits", 0))

cost_saved = max(0.0, (cold_tokens - mem_tokens) / 1000.0 * COST_PER_1K)
time_saved = max(0.0, cold_latency - mem_latency)
token_reduction = (tokens_saved / cold_tokens * 100.0) if cold_tokens else 0.0

c1, c2, c3, c4, c5 = st.columns(5)
odometer(c1, "Tokens Saved", tokens_saved, "od_tokens", color=GOOD_COLOR)
odometer(c2, "Cost Saved (proj $)", cost_saved, "od_cost", prefix="$", decimals=4, color=GOOD_COLOR)
odometer(c3, "Seconds Saved", time_saved, "od_time", decimals=1, color=MEM_COLOR)
odometer(c4, "LLM Calls Saved", max(0, cold_calls - mem_calls), "od_calls", color=MEM_COLOR)
odometer(c5, "Token Reduction %", token_reduction, "od_pct", decimals=1, color=GOOD_COLOR)

def _short(n: float) -> str:
    n = float(n)
    if abs(n) < 1000:
        return f"{n:.0f}"
    if abs(n) < 1_000_000:
        return f"{n/1000:.1f}k"
    return f"{n/1_000_000:.2f}M"


m1, m2, m3, m4 = st.columns(4)
m1.metric("Tasks", tasks)
m2.metric("Memory Hits", memory_hits)
m3.metric("Fast / Deep", f"{totals.get('fast_path_count',0)} / {totals.get('deep_path_count',0)}")
m4.metric("Memory tokens", _short(mem_tokens), delta=f"-{_short(cold_tokens - mem_tokens)} vs cold")

st.markdown("---")

if df.empty:
    st.info("Telemetry initialized, waiting for first event.")
    st.stop()


# --- the race: cumulative cost, memory vs cold -----------------------------
def cumulative(frame: pd.DataFrame, value_col: str) -> pd.Series:
    order = "pair_idx" if "pair_idx" in frame else "timestamp"
    return frame.sort_values(order)[value_col].cumsum().reset_index(drop=True)


def race_chart(title: str, value_col: str, scale: float, yaxis: str) -> go.Figure:
    fig = go.Figure()
    has_variants = "variant" in df.columns
    if has_variants:
        mem = df[df["variant"] == "memory"]
        cold = df[df["variant"] == "no_memory"]
        cold_cum = cumulative(cold, value_col) * scale
        mem_cum = cumulative(mem, value_col) * scale
        x_cold = list(range(1, len(cold_cum) + 1))
        x_mem = list(range(1, len(mem_cum) + 1))
        fig.add_trace(go.Scatter(
            x=x_cold, y=cold_cum, name="Cold crew", mode="lines",
            line=dict(color=COLD_COLOR, width=3)))
        fig.add_trace(go.Scatter(
            x=x_mem, y=mem_cum, name="Memory crew", mode="lines",
            fill="tonexty", fillcolor="rgba(52,211,153,0.15)",
            line=dict(color=MEM_COLOR, width=3)))
    else:
        cum = cumulative(df, value_col) * scale
        fig.add_trace(go.Scatter(
            x=list(range(1, len(cum) + 1)), y=cum, name="Actual",
            mode="lines", line=dict(color=MEM_COLOR, width=3)))
    for marker in amnesia_markers:
        fig.add_vline(
            x=float(marker.get("pair_idx", 0)),
            line_dash="dash", line_color="#f59e0b", line_width=2,
            annotation_text="🧠 wiped", annotation_position="top",
            annotation_font_color="#f59e0b",
        )
    fig.update_layout(
        title=title, template="plotly_dark", height=340,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.12), xaxis_title="task #", yaxis_title=yaxis,
    )
    return fig


r1, r2 = st.columns(2)
with r1:
    st.plotly_chart(race_chart("💸 Cumulative Cost — Memory vs Cold", "token_est_actual",
                               COST_PER_1K / 1000.0, "USD (proj)"), use_container_width=True)
with r2:
    st.plotly_chart(race_chart("⚡ Cumulative Tokens — Memory vs Cold", "token_est_actual",
                               1.0, "tokens"), use_container_width=True)

if "latency_sec" in df.columns:
    st.plotly_chart(race_chart("⏱️ Cumulative Wall-Clock — Memory vs Cold", "latency_sec",
                               1.0, "seconds"), use_container_width=True)


# --- pulsing memory graph --------------------------------------------------
st.subheader("🧠 The shared brain lighting up")
mem_events = df[df["variant"] == "memory"] if "variant" in df.columns else df
if not mem_events.empty and "task_id" in mem_events.columns:
    grp = mem_events.groupby("task_id").agg(
        runs=("task_id", "size"),
        hits=("memory_hits", "sum"),
        last_idx=("pair_idx", "max") if "pair_idx" in mem_events else ("timestamp", "max"),
    ).reset_index()
    recent = grp["last_idx"].max()
    n = len(grp)
    fig = go.Figure()
    cx, cy, R = 0.0, 0.0, 1.0
    xs, ys, sizes, colors, texts = [], [], [], [], []
    for i, row in grp.iterrows():
        ang = 2 * math.pi * i / max(1, n)
        x, y = cx + R * math.cos(ang), cy + R * math.sin(ang)
        xs.append(x); ys.append(y)
        is_recent = row["last_idx"] == recent
        sizes.append(28 + 6 * float(row["hits"]) + (26 if is_recent else 0))
        colors.append(GOOD_COLOR if is_recent else MEM_COLOR)
        texts.append(f"{row['task_id']}<br>hits={int(row['hits'])} runs={int(row['runs'])}")
        # edge to the recently-hit node to suggest retrieval flow
    # hub node
    fig.add_trace(go.Scatter(x=[cx], y=[cy], mode="markers+text", text=["Mem0"],
                             textposition="middle center",
                             marker=dict(size=46, color="#1e293b",
                                         line=dict(color=MEM_COLOR, width=2)),
                             hoverinfo="skip", showlegend=False))
    for x, y in zip(xs, ys):
        fig.add_trace(go.Scatter(x=[cx, x], y=[cy, y], mode="lines",
                                 line=dict(color="rgba(34,211,238,0.25)", width=1),
                                 hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", text=texts, hoverinfo="text",
                             marker=dict(size=sizes, color=colors,
                                         line=dict(color="#0b1020", width=2)),
                             showlegend=False))
    fig.update_layout(template="plotly_dark", height=420,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False, range=[-1.6, 1.6]),
                      yaxis=dict(visible=False, range=[-1.6, 1.6], scaleanchor="x"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Each node is a stored deliverable. The brightest node is the one just retrieved.")

# --- consistency (it remembered) ------------------------------------------
recall = df[df.get("task_id", "") == "recall_probe"] if "task_id" in df.columns else pd.DataFrame()
if not recall.empty and "consistency" in recall.columns:
    st.subheader("🎯 'It remembered' — consistency with the earlier decision")
    cc1, cc2 = st.columns(2)
    mem_c = recall[recall["variant"] == "memory"]["consistency"].max() if "variant" in recall else 0
    cold_c = recall[recall["variant"] == "no_memory"]["consistency"].max() if "variant" in recall else 0
    cc1.metric("Memory crew consistency", f"{mem_c:.0%}")
    cc2.metric("Cold crew consistency", f"{cold_c:.0%}", delta=f"{(cold_c-mem_c):.0%}")

# --- event ticker ----------------------------------------------------------
st.subheader("Latest task events")
cols = [c for c in ["pair_idx", "iteration", "framework", "task_id", "variant", "path",
                    "memory_hits", "llm_requests", "token_est_actual", "latency_sec"]
        if c in df.columns]
st.dataframe(df[cols].tail(16).iloc[::-1], use_container_width=True, hide_index=True)
