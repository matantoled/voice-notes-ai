import os, math, tempfile, time, json, shutil
from datetime import datetime
import numpy as np
import streamlit as st
from faster_whisper import WhisperModel
from ui_style import inject_css, pill, kpi
inject_css()


# Embeddings for semantic search (.npz)
from sentence_transformers import SentenceTransformer

# ---------- NEW: summarization & NLP ----------
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import re

# ---------- Constants ----------
OUTPUT_DIR = "outputs"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

st.set_page_config(page_title="Local AI Transcript Studio", layout="wide")
st.title("🎙️ Local AI Transcript Studio")
st.caption("Local AI transcription with semantic search & summaries (no cloud)")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    model_size = st.selectbox("Model", ["tiny", "base", "small", "medium", "large-v3"], index=2)
    compute_type = st.selectbox("Compute type (CPU)", ["int8", "int8_float32", "float32"], index=0)
    language_hint = st.selectbox("Language hint", ["auto", "en", "he", "ar", "ru", "fr", "es"], index=0)
    use_vad = st.toggle("VAD (remove silence/noise)", value=True)

# ---------- Caches ----------
@st.cache_resource
def load_model(size: str, ctype: str):
    return WhisperModel(size, device="cpu", compute_type=ctype)

@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)

# ---------- Small helpers ----------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_uploaded_to_temp(uploaded) -> str:
    suffix = os.path.splitext(uploaded.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.close()
    return tmp.name

def s_to_timestamp(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    ms = int((s - math.floor(s)) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

def segments_to_srt(segs) -> str:
    lines = []
    for i, g in enumerate(segs, start=1):
        lines.append(
            f"{i}\n{s_to_timestamp(g['start'])} --> {s_to_timestamp(g['end'])}\n{g['text'].strip()}\n"
        )
    return "\n".join(lines)

# ---------- NLTK bootstrap ----------
def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

# ---------- Summarization & Action Items ----------
def summarize_text(text: str, max_sentences: int = 5) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    _ensure_nltk()
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        sent_count = min(max_sentences, max(1, len(parser.document.sentences) // 3 or 1))
        summary_sents = summarizer(parser.document, sent_count)
        summary = " ".join(str(s) for s in summary_sents).strip()
        if not summary:
            summary = " ".join(str(s) for s in parser.document.sentences[:max_sentences]).strip()
        return summary
    except Exception:
        return " ".join(text.split(". ")[:max_sentences]).strip()

def extract_action_items(text: str) -> list[str]:
    _ensure_nltk()
    sents = nltk.sent_tokenize(text or "")
    patterns = [
        r"^(let's|please)\b",
        r"\bwe\s+(need|should|must)\b",
        r"\b(i|we|you)\s+will\b",
        r"\b(todo|follow\s*up|schedule|prepare|create|fix|update|review|send)\b",
        r"\bby\s+(tomorrow|monday|tuesday|wednesday|thursday|friday|next week|\d{1,2}/\d{1,2})\b",
    ]
    rx = re.compile("|".join(patterns), re.IGNORECASE)
    items = [s.strip() for s in sents if rx.search(s)]
    seen, out = set(), []
    for s in items:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out[:20]

# ---------- Persist results to outputs/ ----------
def unique_base(now_str: str) -> str:
    """Create a unique base like 20250910-223045, add -2/-3 if exists."""
    base = now_str
    i = 2
    while os.path.exists(os.path.join(OUTPUT_DIR, f"{base}.json")):
        base = f"{now_str}-{i}"
        i += 1
    return base

def save_all_outputs(
    uploaded, audio_tmp_path: str, segments: list[dict], transcript_text: str,
    srt_text: str, info, model_size: str, compute_type: str, use_vad: bool
) -> dict:
    """
    Save TXT/SRT/JSON + copy audio + save .npz embeddings for semantic search.
    Returns dict with paths.
    """
    ensure_dir(OUTPUT_DIR)

    # base name
    now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    now_str = time.strftime("%Y%m%d-%H%M%S")
    base = unique_base(now_str)

    # file paths
    txt_path = os.path.join(OUTPUT_DIR, f"{base}.txt")
    srt_path = os.path.join(OUTPUT_DIR, f"{base}.srt")
    json_path = os.path.join(OUTPUT_DIR, f"{base}.json")
    npz_path = os.path.join(OUTPUT_DIR, f"{base}.npz")

    # copy audio beside outputs (optional but useful for Search snippet playback)
    audio_ext = os.path.splitext(uploaded.name)[1] or os.path.splitext(audio_tmp_path)[1]
    audio_name = f"{base}{audio_ext}"
    audio_out = os.path.join(OUTPUT_DIR, audio_name)
    try:
        shutil.copyfile(audio_tmp_path, audio_out)
        audio_saved = audio_name  # relative
    except Exception:
        audio_saved = ""

    # write TXT/SRT
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcript_text.strip() + "\n")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_text)

    # write JSON meta (Dashboard/Recent rely on this)
    meta = {
        "base": base,
        "saved_at": now_iso,
        "source_file": uploaded.name,
        "audio_saved": audio_saved,
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "model": model_size,
        "compute_type": compute_type,
        "vad": bool(use_vad),
        "segments": segments,  # [{start,end,text}, ...]
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # embeddings (.npz) for /Search
    try:
        embedder = get_embedder()
        texts = [s["text"] for s in segments]
        starts = np.array([float(s["start"]) for s in segments], dtype=np.float32)
        ends = np.array([float(s["end"]) for s in segments], dtype=np.float32)
        if len(texts) > 0:
            emb = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
            np.savez(npz_path, emb=emb, starts=starts, ends=ends, texts=np.array(texts, dtype=object))
    except Exception as e:
        st.warning(f"Embeddings save failed (ok to ignore): {e}")

    return {
        "base": base,
        "txt": txt_path,
        "srt": srt_path,
        "json": json_path,
        "audio": audio_out if os.path.exists(audio_out) else "",
        "npz": npz_path if os.path.exists(npz_path) else "",
    }

# ---------- UI: upload ----------
uploaded = st.file_uploader(
    "Upload an audio file",
    type=["mp3", "wav", "m4a", "flac", "ogg"]
)

if uploaded:
    st.audio(uploaded)
else:
    st.info("Choose a file to begin.")

# ---------- Action ----------
start = st.button("Start transcription", disabled=(uploaded is None))

if start:
    if uploaded is None:
        st.warning("Please upload an audio file first.")
        st.stop()

    audio_path = save_uploaded_to_temp(uploaded)

    try:
        with st.spinner("Loading model... (first time can take a bit)"):
            model = load_model(model_size, compute_type)

        with st.spinner("Transcribing..."):
            segments_gen, info = model.transcribe(
                audio_path,
                vad_filter=use_vad,
                language=None if language_hint == "auto" else language_hint
            )
            segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segments_gen]

        st.success(f"Done. Detected language: {info.language} (p={info.language_probability:.2f})")
        transcript_text = "\n".join(s["text"].strip() for s in segments)

        with st.expander("Transcript"):
            st.text_area("Text", transcript_text, height=220)

        with st.expander("Timestamps"):
            for i, s in enumerate(segments, start=1):
                st.write(f"{i:03d} [{s_to_timestamp(s['start'])} → {s_to_timestamp(s['end'])}] {s['text']}")

        # Build SRT & downloads
        srt_text = segments_to_srt(segments)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇ TXT", transcript_text.encode("utf-8"), "transcript.txt")
        with c2:
            st.download_button("⬇ SRT", srt_text.encode("utf-8"), "transcript.srt")

        # ---------- Auto summary + Action items ----------
        with st.spinner("Summarizing (local)..."):
            summary = summarize_text(transcript_text, max_sentences=5)
            actions = extract_action_items(transcript_text)

        with st.expander("Auto-summary"):
            if summary:
                st.write(summary)
                st.download_button("⬇ Summary (TXT)", summary.encode("utf-8"), "summary.txt")
            else:
                st.info("No summary available for this audio.")

        with st.expander("Action items"):
            if actions:
                for i, line in enumerate(actions, start=1):
                    st.write(f"{i}. {line}")
                actions_txt = "\n".join(f"- {a}" for a in actions)
                st.download_button("⬇ Action items (TXT)", actions_txt.encode("utf-8"), "action_items.txt")
            else:
                st.info("No action items found.")

        # ---------- SAVE EVERYTHING to outputs/ ----------
        with st.spinner("Saving to outputs/…"):
            saved = save_all_outputs(
                uploaded=uploaded,
                audio_tmp_path=audio_path,
                segments=segments,
                transcript_text=transcript_text,
                srt_text=srt_text,
                info=info,
                model_size=model_size,
                compute_type=compute_type,
                use_vad=use_vad,
            )

        msg = f"Saved: {os.path.relpath(saved['txt'])} • {os.path.relpath(saved['srt'])} • {os.path.relpath(saved['json'])}"
        if saved.get("npz"):
            msg += f" • {os.path.relpath(saved['npz'])}"
        st.info(msg)

    except Exception as e:
        st.error(f"Transcription failed: {e}")
    finally:
        try:
            os.remove(audio_path)
        except Exception:
            pass
