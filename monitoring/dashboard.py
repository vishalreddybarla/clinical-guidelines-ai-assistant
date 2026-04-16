"""Streamlit monitoring dashboard — displays request logs and usage stats."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from project root without installing as package
ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.monitoring import get_all_logs

st.set_page_config(
    page_title="Clinical Guidelines — Monitoring",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Clinical Guidelines AI — Monitoring Dashboard")

logs = get_all_logs()

if not logs:
    st.info("No requests logged yet. Use the chat interface to generate some queries.")
    st.stop()

df = pd.DataFrame(logs)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.date

# --- KPI row ---
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Requests", len(df))
col2.metric("Avg Latency (ms)", f"{df['latency_ms'].mean():.0f}")
col3.metric("Total Cost (USD)", f"${df['cost_usd'].sum():.4f}")
col4.metric("Avg Tokens / Request", f"{df[['tokens_in','tokens_out']].sum(axis=1).mean():.0f}")
rated = df[df["user_rating"].notna()]
avg_rating = rated["user_rating"].mean() if not rated.empty else 0.0
col5.metric("Avg User Rating", f"{avg_rating:.1f} / 5")

st.divider()

# --- Charts row ---
chart_col1, chart_col2, chart_col3 = st.columns(3)

with chart_col1:
    st.subheader("Requests per day")
    daily = df.groupby("date").size().reset_index(name="count")
    st.bar_chart(daily.set_index("date")["count"])

with chart_col2:
    st.subheader("Model usage")
    model_counts = df["model"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(model_counts.values, labels=model_counts.index, autopct="%1.1f%%")
    st.pyplot(fig)
    plt.close(fig)

with chart_col3:
    st.subheader("Tool usage")
    # Explode the comma-separated tools_used column
    tool_series = df["tools_used"].dropna().str.split(",").explode().str.strip()
    if not tool_series.empty:
        tool_counts = tool_series.value_counts()
        st.bar_chart(tool_counts)
    else:
        st.info("No tool usage data yet.")

st.divider()

# --- Latency histogram ---
st.subheader("Latency distribution (ms)")
fig, ax = plt.subplots(figsize=(8, 2.5))
ax.hist(df["latency_ms"].dropna(), bins=30, edgecolor="black")
ax.set_xlabel("Latency (ms)")
ax.set_ylabel("Count")
st.pyplot(fig)
plt.close(fig)

st.divider()

# --- User ratings ---
if not rated.empty:
    st.subheader("User ratings distribution")
    rating_counts = rated["user_rating"].value_counts().sort_index()
    st.bar_chart(rating_counts)

st.divider()

# --- Recent requests table ---
st.subheader("Recent requests")
display_cols = ["timestamp", "query", "model", "tokens_in", "tokens_out", "latency_ms", "cost_usd", "user_rating"]
available = [c for c in display_cols if c in df.columns]
st.dataframe(
    df[available].head(100).sort_values("timestamp", ascending=False),
    use_container_width=True,
    hide_index=True,
)
