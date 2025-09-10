import os, glob, json, re
import pandas as pd
import streamlit as st
from ui_style import inject_css, pill, kpi
inject_css()


# ---------- page setup ----------
st.set_page_config(page_title="Recent transcripts", layout="wide")
st.title("🗂️ Recent transcripts")

OUTPUT_DIR = "outputs"

# ---------- optional TextRank (sumy) ----------
_SUMY_OK = True
try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer
except Exception:
    _SUMY_OK = False

# ---------- helpers: io/delete ----------
def safe_remove(path: str):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except Exception as e:
        st.error(f"Delete failed for {path}: {e}")

def delete_record(base: str):
    safe_remove(os.path.join(OUTPUT_DIR, f"{base}.json"))
    safe_remove(os.path.join(OUTPUT_DIR, f"{base}.txt"))
    safe_remove(os.path.join(OUTPUT_DIR, f"{base}.srt"))
    # optional extra files
    for ext in (".npz", ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".summary.txt", ".action.txt"):
        p = os.path.join(OUTPUT_DIR, f"{base}{ext}")
        if os.path.exists(p):
            safe_remove(p)

if not os.path.isdir(OUTPUT_DIR):
    st.info("No outputs yet. Run a transcription first.")
    st.stop()

# ---------- helpers: transcript text & summary ----------
def _read_txt(base: str) -> tuple[str, float] | tuple[None, None]:
    """Return (text, mtime) from outputs/<base>.txt if exists."""
    p = os.path.join(OUTPUT_DIR, f"{base}.txt")
    if not os.path.exists(p):
        return None, None
    try:
        with open(p, encoding="utf-8", errors="ignore") as f:
            return f.read(), os.path.getmtime(p)
    except Exception:
        return None, None

def _read_json_text(base: str) -> tuple[str, float] | tuple[None, None]:
    """Fallback: build text from outputs/<base>.json segments."""
    p = os.path.join(OUTPUT_DIR, f"{base}.json")
    if not os.path.exists(p):
        return None, None
    try:
        with open(p, encoding="utf-8") as f:
            meta = json.load(f)
        segs = meta.get("segments", []) or []
        text = "\n".join(s.get("text", "") for s in segs)
        return text, os.path.getmtime(p)
    except Exception:
        return None, None

def transcript_text_for_base(base: str) -> tuple[str, float] | tuple[None, None]:
    """Prefer TXT; otherwise JSON; return (text, source_mtime)."""
    txt, mt = _read_txt(base)
    if txt is not None:
        return txt, mt
    return _read_json_text(base)

def _summarize_textrank(text: str, sentences: int = 6) -> str:
    """Use sumy TextRank if available; fallback to first-N sentences."""
    if not text.strip():
        return ""
    if _SUMY_OK:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, sentences)
        return "\n".join(str(s) for s in summary).strip()
    # Fallback: take first-N sentences
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:sentences]).strip()

def _extract_action_items(text: str, limit: int = 10) -> list[str]:
    """Very light heuristic to extract to-dos / action-like lines."""
    sents = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    rx = re.compile(
        r"\b(action items?|todo|follow[- ]?up|next steps?|we (need|should)|let'?s|please|assign|schedule|send|review)\b",
        re.IGNORECASE,
    )
    hits = [s.strip() for s in sents if rx.search(s)]
    if not hits:
        rx2 = re.compile(r"\b(need to|should|let'?s|please)\b", re.IGNORECASE)
        hits = [s.strip() for s in sents if rx2.search(s)]
    return hits[:limit]

def _summary_paths(base: str) -> tuple[str, str]:
    return (
        os.path.join(OUTPUT_DIR, f"{base}.summary.txt"),
        os.path.join(OUTPUT_DIR, f"{base}.action.txt"),
    )

@st.cache_data(show_spinner=False)
def get_summary_for_base(base: str, mtime_key: int) -> tuple[str, list[str]]:
    """
    Return (summary, action_items). Will (re)compute if files missing or older than source.
    mtime_key is an int key we pass from the caller to bust cache if source changed.
    """
    text, src_mtime = transcript_text_for_base(base)
    if not text:
        return "", []

    sum_path, act_path = _summary_paths(base)
    need = True
    if os.path.exists(sum_path) and os.path.exists(act_path):
        try:
            need = os.path.getmtime(sum_path) < src_mtime or os.path.getmtime(act_path) < src_mtime
        except Exception:
            need = True

    if need:
        trimmed = text if len(text) <= 100_000 else text[:100_000]
        summary = _summarize_textrank(trimmed, sentences=6) or "(no summary)"
        actions = _extract_action_items(trimmed, limit=10)
        try:
            with open(sum_path, "w", encoding="utf-8") as f:
                f.write(summary.strip() + "\n")
            with open(act_path, "w", encoding="utf-8") as f:
                f.write(("\n".join(f"- {a}" for a in actions) or "- (none)") + "\n")
        except Exception:
            pass
        return summary, actions
    else:
        try:
            with open(sum_path, encoding="utf-8", errors="ignore") as f:
                summary = f.read().strip()
        except Exception:
            summary = "(no summary)"
        try:
            with open(act_path, encoding="utf-8", errors="ignore") as f:
                actions = [ln.strip("- ").strip() for ln in f.read().splitlines() if ln.strip()]
        except Exception:
            actions = []
        return summary, actions

# ---------- UI ----------
q = st.text_input("Search (simple contains match)", "")

paths = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.json")), reverse=True)
shown = 0

for path in paths:
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)

    base = os.path.splitext(os.path.basename(path))[0]
    segs = meta.get("segments", [])
    text_preview = "\n".join(s.get("text", "") for s in segs)

    if q and q.lower() not in text_preview.lower():
        continue

    shown += 1
    duration = segs[-1]["end"] if segs else 0.0
    words = sum(len(s.get("text","").split()) for s in segs)
    wpm = (words / (duration/60.0)) if duration > 0 else 0.0

    with st.expander(f"{base} — {meta.get('language','?').upper()} — {meta.get('source_file','')}"):
        st.caption(
            f"Saved: {meta.get('saved_at')} • Model: {meta.get('model')} "
            f"({meta.get('compute_type')}) • VAD: {meta.get('vad')}"
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Duration", f"{duration/60:.1f} min")
        m2.metric("Segments", f"{len(segs)}")
        m3.metric("WPM", f"{wpm:.0f}")

        st.text_area("Preview", text_preview[:4000], height=160, key=f"preview_{base}")

        # Optional mini chart
        show_chart = st.checkbox("Show segment length chart", value=False, key=f"chart_{base}")
        if show_chart and segs:
            import pandas as _pd  # local import to avoid top-level cost if unused
            df = _pd.DataFrame({
                "segment": list(range(1, len(segs)+1)),
                "duration_sec": [max(0.0, s["end"]-s["start"]) for s in segs]
            })
            st.bar_chart(df, x="segment", y="duration_sec", height=140)

        col1, col2, col3, col4 = st.columns([1,1,1,1])
        with col1:
            txt_path = os.path.join(OUTPUT_DIR, f"{base}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, "rb") as f:
                    st.download_button("⬇ TXT", f.read(), file_name=f"{base}.txt", key=f"dl_txt_{base}")
        with col2:
            srt_path = os.path.join(OUTPUT_DIR, f"{base}.srt")
            if os.path.exists(srt_path):
                with open(srt_path, "rb") as f:
                    st.download_button("⬇ SRT", f.read(), file_name=f"{base}.srt", key=f"dl_srt_{base}")
        with col3:
            aud_rel = meta.get("audio_saved")
            if aud_rel:
                full = os.path.join(OUTPUT_DIR, aud_rel)
                if os.path.exists(full):
                    with open(full, "rb") as f:
                        st.download_button("⬇ Audio", f.read(), file_name=os.path.basename(full), key=f"dl_audio_{base}")
        with col4:
            with st.form(key=f"del_form_{base}", clear_on_submit=True):
                confirm = st.checkbox("Confirm delete", key=f"confirm_{base}")
                submitted = st.form_submit_button("🗑 Delete")
                if submitted:
                    if confirm:
                        delete_record(base)
                        try:
                            st.experimental_rerun()
                        except Exception:
                            st.rerun()

        # -------- Summary & Action items (available here too) --------
        st.markdown("---")
        st.subheader("Summary & action items")
        # use transcript mtime (from TXT if exists, else JSON)
        _, src_mtime = transcript_text_for_base(base)
        mtime_key = int(src_mtime or 0)
        with st.spinner("Loading..."):
            summary, actions = get_summary_for_base(base, mtime_key)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Summary**")
            st.write(summary or "(no summary)")
            st.download_button(
                "⬇ Summary (TXT)",
                (summary or "(no summary)").encode("utf-8"),
                file_name=f"{base}.summary.txt",
                key=f"dl_sum_{base}",
            )
        with c2:
            st.markdown("**Action items**")
            if actions:
                st.markdown("\n".join(f"- {a}" for a in actions))
            else:
                st.write("(none)")
            st.download_button(
                "⬇ Action items (TXT)",
                ("\n".join(f"- {a}" for a in actions) or "- (none)").encode("utf-8"),
                file_name=f"{base}.action.txt",
                key=f"dl_act_{base}",
            )

if shown == 0:
    st.info("No results match your search.")
