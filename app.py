import os, math, tempfile
import streamlit as st
from faster_whisper import WhisperModel

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
