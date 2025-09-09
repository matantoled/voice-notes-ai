import os, glob, json
from collections import Counter
from datetime import datetime

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("📊 Dashboard")

OUTPUT_DIR = "outputs"

if not os.path.isdir(OUTPUT_DIR):
    st.info("No outputs yet. Transcribe something first.")
    st.stop()

paths = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.json")), reverse=True)
if not paths:
    st.info("No saved transcripts yet.")
    st.stop()

records = []
for p in paths:
    with open(p, encoding="utf-8") as f:
        meta = json.load(f)
    segments = meta.get("segments", [])
    duration = segments[-1]["end"] if segments else 0.0
    words = sum(len(s.get("text","").split()) for s in segments)
    records.append({
        "base": os.path.splitext(os.path.basename(p))[0],
        "saved_at": meta.get("saved_at"),
        "date": meta.get("saved_at","")[:10],
        "language": (meta.get("language") or "").upper(),
        "model": meta.get("model"),
        "compute_type": meta.get("compute_type"),
        "vad": meta.get("vad"),
        "duration_min": duration / 60.0,
        "segments": len(segments),
        "words": words
    })

df = pd.DataFrame(records)

# Top metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transcripts", len(df))
c2.metric("Total minutes", f"{df['duration_min'].sum():.1f}")
c3.metric("Total words", f"{int(df['words'].sum())}")
avg_wpm = (df["words"].sum() / (df["duration_min"].sum() + 1e-9))
c4.metric("Avg WPM", f"{avg_wpm:.0f}")

# Charts
lc, rc = st.columns(2)
with lc:
    st.subheader("Minutes by day")
    by_day = df.groupby("date")["duration_min"].sum().reset_index()
    st.line_chart(by_day, x="date", y="duration_min", height=220)

with rc:
    st.subheader("Language distribution")
    lang = df["language"].value_counts().reset_index()
    lang.columns = ["language", "count"]
    st.bar_chart(lang, x="language", y="count", height=220)

st.subheader("All transcripts")
st.dataframe(df[["base","date","language","model","compute_type","duration_min","segments","words"]], use_container_width=True, height=260)
