import os, glob, json, re
from datetime import datetime, date
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

# Optional wordcloud (nice-to-have)
try:
    from wordcloud import WordCloud
    WORDCLOUD_OK = True
except Exception:
    WORDCLOUD_OK = False

import altair as alt
alt.data_transformers.disable_max_rows()

st.set_page_config(page_title="Dashboard", layout="wide")

# ------------------ helpers ------------------
OUTPUT_DIR = "outputs"

def load_records() -> pd.DataFrame:
    """Read all saved JSON metadata files and compute per-transcript metrics."""
    paths = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.json")))
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            meta = json.load(f)

        segs = meta.get("segments", [])
        if not isinstance(segs, list):
            continue

        duration = segs[-1]["end"] if segs else 0.0
        speech_time = float(np.sum([max(0.0, s["end"] - s["start"]) for s in segs])) if segs else 0.0
        text_all = " ".join(s.get("text", "") for s in segs)
        words = len(text_all.split())
        wpm = (words / (duration / 60.0)) if duration > 0 else 0.0
        silence_ratio = 1.0 - (speech_time / duration) if duration > 0 else 0.0

        base = os.path.splitext(os.path.basename(p))[0]
        saved = meta.get("saved_at") or datetime.now().isoformat()
        try:
            dt = datetime.fromisoformat(saved.replace("Z", ""))
        except Exception:
            dt = datetime.now()

        rows.append(
            {
                "base": base,
                "date": dt.date(),
                "datetime": dt,
                "year": dt.year,
                "month": dt.strftime("%Y-%m"),
                "weekday": dt.strftime("%a"),
                "hour": dt.hour,
                "language": (meta.get("language") or "").upper(),
                "model": meta.get("model"),
                "compute_type": meta.get("compute_type"),
                "vad": bool(meta.get("vad")),
                "duration_min": duration / 60.0,
                "speech_min": speech_time / 60.0,
                "silence_ratio": silence_ratio,
                "segments": len(segs),
                "words": words,
                "wpm": wpm,
                "text": text_all,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "base","date","datetime","year","month","weekday","hour",
                "language","model","compute_type","vad",
                "duration_min","speech_min","silence_ratio","segments","words","wpm","text",
            ]
        )

    df = (
        pd.DataFrame(rows)
        .sort_values("datetime", ascending=False)
        .reset_index(drop=True)
    )
    return df


# simple tokenizer (Unicode-friendly) + minimal English stopwords
STOP = set(
    """
the a an and or of to in for on with that this is are was were be been from at by as it you your we i me my our us they them their his her its not no yes do did done
""".split()
)

def tokenize(text: str):
    toks = re.split(r"[^0-9A-Za-z\u00C0-\uFFFF]+", text.lower())
    return [t for t in toks if t and t not in STOP and len(t) > 2]

def top_keywords(df: pd.DataFrame, k: int = 30) -> pd.DataFrame:
    bag = Counter()
    for t in df["text"]:
        bag.update(tokenize(t))
    return pd.DataFrame(bag.most_common(k), columns=["token", "count"])


# ------------------ data ------------------
st.title("📊 Dashboard")
if not os.path.isdir(OUTPUT_DIR):
    st.info("No outputs yet. Transcribe something first on the main page.")
    st.stop()

df = load_records()
if df.empty:
    st.info("No saved transcripts yet.")
    st.stop()

# ------------------ filters ------------------
min_date, max_date = df["date"].min(), df["date"].max()
with st.sidebar:
    st.header("Filters")
    dr = st.date_input(
        "Date range",
        (min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    langs = st.multiselect(
        "Languages",
        sorted(df["language"].unique().tolist()),
        default=sorted(df["language"].unique()),
    )
    models = st.multiselect(
        "Models",
        sorted(df["model"].dropna().unique().tolist()),
        default=sorted(df["model"].dropna().unique()),
    )
    show_vad = st.selectbox("VAD", ["All", "Only VAD", "Only No-VAD"], index=0)

# normalize dates
if isinstance(dr, tuple) and len(dr) == 2:
    d_from, d_to = dr
else:
    d_from, d_to = (dr, dr)
if not isinstance(d_from, date):
    d_from = pd.to_datetime(d_from).date()
if not isinstance(d_to, date):
    d_to = pd.to_datetime(d_to).date()

mask = (df["date"] >= d_from) & (df["date"] <= d_to)
if langs:
    mask &= df["language"].isin(langs)
if models:
    mask &= df["model"].isin(models)
if show_vad == "Only VAD":
    mask &= df["vad"] == True
elif show_vad == "Only No-VAD":
    mask &= df["vad"] == False

fdf = df.loc[mask].copy()
if fdf.empty:
    st.warning("No data after filters — try widening the range.")
    st.stop()

# ------------------ KPIs ------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Transcripts", f"{len(fdf)}")
c2.metric("Total minutes", f"{fdf['duration_min'].sum():.1f}")
c3.metric("Total words", f"{int(fdf['words'].sum())}")
c4.metric("Avg WPM", f"{(fdf['words'].sum()/(fdf['duration_min'].sum()+1e-9)):.0f}")
c5.metric("Avg silence", f"{(fdf['silence_ratio'].mean()*100):.0f}%")

# ------------------ highlights ------------------
hl1 = fdf.loc[fdf["duration_min"].idxmax()]
hl2 = fdf.loc[fdf["wpm"].idxmax()]
hl3 = fdf.loc[fdf["segments"].idxmax()]
hl4 = fdf.loc[fdf["silence_ratio"].idxmin()]

st.markdown("### ⭐ Highlights")
h1, h2, h3, h4 = st.columns(4)
h1.markdown(f"**Longest**: {hl1['base']}  \n{hl1['duration_min']:.1f} min")
h2.markdown(f"**Fastest speaker**: {hl2['base']}  \n{hl2['wpm']:.0f} WPM")
h3.markdown(f"**Most segments**: {hl3['base']}  \n{hl3['segments']}")
h4.markdown(f"**Lowest silence**: {hl4['base']}  \n{(hl4['silence_ratio']*100):.0f}%")

st.markdown("---")

# ------------------ row 1: Top durations (no time dependency) + Languages ------------------
lc, rc = st.columns([2, 1])

with lc:
    st.subheader("Top transcripts by duration (min)")

    # Safe bounds for slider (works with tiny datasets)
    n_total = len(fdf)
    n_max = max(1, min(20, n_total))      # at least 1, at most 20
    default = min(10, n_max)              # default within bounds

    if n_max == 1:
        top_n = 1
        st.caption("Showing top 1 (only one transcript).")
    else:
        top_n = st.slider("Top-N", 1, n_max, default, key="topn_duration")

    topd = (
        fdf.nlargest(top_n, "duration_min")[["base", "duration_min"]]
        .iloc[::-1]  # reverse for nicer order (longest at top)
    )

    bar = alt.Chart(topd).mark_bar().encode(
        x=alt.X("duration_min:Q", title="Minutes"),
        y=alt.Y("base:N", title="Transcript", sort=None),
        tooltip=[
            alt.Tooltip("base:N", title="Transcript"),
            alt.Tooltip("duration_min:Q", title="Minutes", format=".1f"),
        ],
    )
    st.altair_chart(bar, use_container_width=True)

with rc:
    st.subheader("Languages")
    lang = fdf["language"].value_counts().reset_index()
    lang.columns = ["language", "count"]
    chart = alt.Chart(lang).mark_arc(innerRadius=60).encode(
        theta="count:Q", color="language:N", tooltip=["language", "count"]
    )
    st.altair_chart(chart, use_container_width=True)

# ------------------ row 2: Quadrant + Scatter ------------------
lc2, rc2 = st.columns(2)

with lc2:
    st.subheader("Silence (%) vs WPM (quadrant)")
    qdf = fdf.assign(silence_pct=(fdf["silence_ratio"] * 100.0))
    med_wpm = float(qdf["wpm"].median()) if not qdf.empty else 0.0
    med_sil = float(qdf["silence_pct"].median()) if not qdf.empty else 0.0

    base = alt.Chart(qdf).encode(
        x=alt.X("wpm:Q", title="WPM"),
        y=alt.Y("silence_pct:Q", title="Silence (%)"),
        color=alt.Color("language:N", title="Lang"),
        tooltip=["base","language","wpm",alt.Tooltip("silence_pct:Q", title="Silence (%)", format=".0f")]
    )
    pts = base.mark_circle(size=90, opacity=0.7)
    rule_x = alt.Chart(pd.DataFrame({"x":[med_wpm]})).mark_rule(strokeDash=[4,4]).encode(x="x:Q")
    rule_y = alt.Chart(pd.DataFrame({"y":[med_sil]})).mark_rule(strokeDash=[4,4]).encode(y="y:Q")
    st.altair_chart(pts + rule_x + rule_y, use_container_width=True)

with rc2:
    st.subheader("Duration vs Words")
    scat = alt.Chart(fdf).mark_circle(size=80, opacity=0.6).encode(
        x=alt.X("duration_min:Q", title="Duration (min)"),
        y=alt.Y("words:Q", title="Words"),
        color=alt.Color("language:N", title="Lang"),
        tooltip=["base", "language", "duration_min", "words", "wpm"],
    )
    st.altair_chart(scat, use_container_width=True)

st.markdown("---")

# ------------------ row 3: Distributions ------------------
dc1, dc2 = st.columns(2)

with dc1:
    st.subheader("WPM distribution")
    hist_wpm = alt.Chart(fdf).mark_bar().encode(
        x=alt.X("bin(wpm, step=10):Q", title="WPM"),
        y=alt.Y("count():Q", title="Count"),
        tooltip=[alt.Tooltip("count():Q", title="Count")]
    )
    st.altair_chart(hist_wpm, use_container_width=True)

with dc2:
    st.subheader("Segments distribution")
    hist_seg = alt.Chart(fdf).mark_bar().encode(
        x=alt.X("bin(segments, step=5):Q", title="Segments"),
        y=alt.Y("count():Q", title="Count"),
        tooltip=[alt.Tooltip("count():Q", title="Count")]
    )
    st.altair_chart(hist_seg, use_container_width=True)

st.markdown("---")

# ------------------ keywords ------------------
st.subheader("Top keywords")
kw = top_keywords(fdf, k=30)
kc1, kc2 = st.columns([1, 1])

with kc1:
    bar_kw = alt.Chart(kw).mark_bar().encode(
        x="count:Q", y=alt.Y("token:N", sort="-x"), tooltip=["token", "count"]
    )
    st.altair_chart(bar_kw, use_container_width=True)

with kc2:
    if WORDCLOUD_OK and not kw.empty:
        wc = WordCloud(width=800, height=300, background_color=None, mode="RGBA")
        wc.generate_from_frequencies(dict(zip(kw["token"], kw["count"])))
        st.image(wc.to_array(), use_container_width=True)
    else:
        st.caption("Install `wordcloud` to see a word cloud (optional).")

st.markdown("---")

# ------------------ table + download ------------------
st.subheader("All transcripts")
show_cols = ["base", "date", "language", "model", "duration_min", "segments", "words", "wpm", "silence_ratio"]
styled = fdf[show_cols].copy()
styled["duration_min"] = styled["duration_min"].round(1)
styled["wpm"] = styled["wpm"].round(0)
styled["silence_ratio"] = (styled["silence_ratio"] * 100).round(0).astype(int).astype(str) + "%"

st.dataframe(styled, use_container_width=True, height=280)

csv = fdf.to_csv(index=False).encode("utf-8")
st.download_button("⬇ Download CSV", data=csv, file_name="dashboard_export.csv")
