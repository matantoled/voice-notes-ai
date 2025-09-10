import os, math, tempfile, time
import streamlit as st
from faster_whisper import WhisperModel

# ---------- NEW: summarization & NLP ----------
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import re

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

# ---------- Helpers ----------
@st.cache_resource
def load_model(size: str, ctype: str):
    # Load once (cached) to speed up subsequent runs
    return WhisperModel(size, device="cpu", compute_type=ctype)

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

# ---------- NEW: NLTK bootstrap ----------
def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

# ---------- NEW: text summarization (local) ----------
def summarize_text(text: str, max_sentences: int = 5) -> str:
    """
    Summarize long text into up to max_sentences sentences (English best).
    Falls back gracefully if text is short or unsupported.
    """
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

# ---------- NEW: naive action-items (local, rule-based) ----------
def extract_action_items(text: str) -> list[str]:
    """
    Simple heuristic: pick sentences that *look* like tasks/requests.
    Works best in English. Tune patterns later as you like.
    """
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

        with st.expander("Transcript"):
            st.text_area("Text", transcript_text, height=220)

        with st.expander("Timestamps"):
            for i, s in enumerate(segments, start=1):
                st.write(f"{i:03d} [{s_to_timestamp(s['start'])} → {s_to_timestamp(s['end'])}] {s['text']}")

        # Downloads (TXT / SRT)
        srt_text = segments_to_srt(segments)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇ TXT", transcript_text.encode("utf-8"), "transcript.txt")
        with c2:
            st.download_button("⬇ SRT", srt_text.encode("utf-8"), "transcript.srt")

        # ---------- NEW: Auto summary + Action items ----------
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

    except Exception as e:
        # Typical case: FFmpeg not installed or not in PATH
        st.error(f"Transcription failed: {e}")
    finally:
        try:
            os.remove(audio_path)
        except Exception:
            pass
