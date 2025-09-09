import os, math, tempfile, json, re, shutil
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer

# ----- constants -----
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # multilingual (he/en)

# ---------- Page setup ----------
st.set_page_config(page_title="Interview Transcriber", layout="wide")
st.title("🎙️ Interview Transcriber")
st.caption("Local transcription with faster-whisper (no cloud)")

# ---------- Sidebar settings ----------
with st.sidebar:
    st.header("Settings")
    model_size = st.selectbox("Model", ["tiny", "base", "small", "medium", "large-v3"], index=2)
    compute_type = st.selectbox("Compute type (CPU)", ["int8", "int8_float32", "float32"], index=0)
    language_hint = st.selectbox("Language hint", ["auto", "en", "he", "ar", "ru", "fr", "es"], index=0)
    use_vad = st.toggle("VAD (remove silence/noise)", value=True)
    save_audio_orig = st.toggle("Save original audio file", value=True)

# ---------- Helpers ----------
@st.cache_resource
def load_model(size: str, ctype: str):
    # Load once (cached) to speed up subsequent runs
    return WhisperModel(size, device="cpu", compute_type=ctype)

@st.cache_resource
def get_embedder():
    # Load embedding model once (cached)
    return SentenceTransformer(EMBED_MODEL)

def save_uploaded_to_temp(uploaded) -> str:
    # Save the uploaded file to a temporary path so the ASR can read it
    suffix = os.path.splitext(uploaded.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.close()
    return tmp.name

def s_to_timestamp(s: float) -> str:
    # Format seconds to SRT timestamp 00:00:00,000
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    ms = int((s - math.floor(s)) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

def segments_to_srt(segs) -> str:
    # Build SRT text from segment dicts
    lines = []
    for i, g in enumerate(segs, start=1):
        lines.append(
            f"{i}\n{s_to_timestamp(g['start'])} --> {s_to_timestamp(g['end'])}\n{g['text'].strip()}\n"
        )
    return "\n".join(lines)

def slugify(name: str) -> str:
    # Safe filename: letters, numbers, dash, underscore
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower()

def save_all(transcript_text: str, segments: list, info, uploaded_name: str,
             model_size: str, compute_type: str, use_vad: bool,
             audio_src_path: str, save_audio_orig: bool) -> str:
    """
    Persist TXT/SRT/JSON under outputs/. Optionally copy original audio.
    Returns base filename (without extension).
    """
    os.makedirs("outputs", exist_ok=True)
    base = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{slugify(os.path.splitext(uploaded_name)[0])}"

    with open(f"outputs/{base}.txt", "w", encoding="utf-8") as f:
        f.write(transcript_text)

    with open(f"outputs/{base}.srt", "w", encoding="utf-8") as f:
        f.write(segments_to_srt(segments))

    audio_saved = None
    if save_audio_orig:
        suffix = os.path.splitext(uploaded_name)[1]
        audio_saved = f"{base}{suffix}"
        try:
            shutil.copyfile(audio_src_path, os.path.join("outputs", audio_saved))
        except Exception:
            audio_saved = None  # don't fail save if copy fails

    meta = {
        "source_file": uploaded_name,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "model": model_size,
        "compute_type": compute_type,
        "vad": use_vad,
        "audio_saved": audio_saved,             # NEW
        "segments": segments,
    }
    with open(f"outputs/{base}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return base

def compute_and_save_embeddings(segments: list, base: str):
    # Build embeddings for each segment and save to outputs/<base>.npz
    texts = [s["text"].strip() for s in segments if s["text"].strip()]
    starts = np.array([s["start"] for s in segments if s["text"].strip()], dtype=np.float32)
    ends   = np.array([s["end"]   for s in segments if s["text"].strip()], dtype=np.float32)
    if len(texts) == 0:
        return False

    embedder = get_embedder()
    emb = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)  # cosine-ready

    np.savez(
        f"outputs/{base}.npz",
        emb=emb.astype(np.float32),
        starts=starts,
        ends=ends,
        texts=np.array(texts, dtype=object),
        embed_model=EMBED_MODEL
    )
    return True

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

    # Save a temp copy of the uploaded file
    audio_path = save_uploaded_to_temp(uploaded)

    try:
        # Load model (first time may be slower due to download/init)
        with st.spinner("Loading model... (first time can take a bit)"):
            model = load_model(model_size, compute_type)

        # Real transcription
        with st.spinner("Transcribing..."):
            segments_gen, info = model.transcribe(
                audio_path,
                vad_filter=use_vad,
                language=None if language_hint == "auto" else language_hint
            )
            segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segments_gen]

        # Display
        st.success(f"Done. Detected language: {info.language} (p={info.language_probability:.2f})")
        transcript_text = "\n".join(s["text"].strip() for s in segments)

        # Basic stats & chart
        total_seconds = segments[-1]["end"] if segments else 0.0
        word_count = len(transcript_text.split())
        wpm = (word_count / (total_seconds / 60.0)) if total_seconds > 0 else 0.0
        seg_durations = [max(0.0, s["end"] - s["start"]) for s in segments]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Segments", f"{len(segments)}")
        m2.metric("Duration", f"{total_seconds/60:.1f} min")
        m3.metric("Words", f"{word_count}")
        m4.metric("WPM", f"{wpm:.0f}")

        if segments:
            df = pd.DataFrame({
                "segment": list(range(1, len(segments) + 1)),
                "duration_sec": seg_durations
            })
            st.caption("Segment durations")
            st.bar_chart(df, x="segment", y="duration_sec", height=160)

        # Persist outputs (TXT/SRT/JSON + optional audio copy)
        base = save_all(
            transcript_text, segments, info, uploaded.name,
            model_size, compute_type, use_vad,
            audio_src_path=audio_path, save_audio_orig=save_audio_orig
        )
        st.info(f"Saved: outputs/{base}.txt · outputs/{base}.srt · outputs/{base}.json" + (f" · outputs/{base}{os.path.splitext(uploaded.name)[1]}" if save_audio_orig else ""))

        # Embeddings
        with st.spinner("Indexing (embeddings)..."):
            ok = compute_and_save_embeddings(segments, base)
        if ok:
            st.success(f"Embeddings saved: outputs/{base}.npz")
        else:
            st.warning("No non-empty segments to index.")

        with st.expander("Transcript"):
            st.text_area("Text", transcript_text, height=220)

        with st.expander("Timestamps"):
            for i, s in enumerate(segments, start=1):
                st.write(f"{i:03d} [{s_to_timestamp(s['start'])} → {s_to_timestamp(s['end'])}] {s['text']}")

        # Downloads
        srt_text = segments_to_srt(segments)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇ TXT", transcript_text.encode("utf-8"), "transcript.txt")
        with c2:
            st.download_button("⬇ SRT", srt_text.encode("utf-8"), "transcript.srt")

    except Exception as e:
        # Typical case: FFmpeg not installed or not in PATH
        st.error(f"Transcription failed: {e}")
    finally:
        try:
            os.remove(audio_path)
        except Exception:
            pass
