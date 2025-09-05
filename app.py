import time
import streamlit as st

# --- Page setup ---
st.set_page_config(page_title="Interview Transcriber", layout="wide")
st.title("🎙️ Interview Transcriber")
st.caption("MVP: UI only (no AI wired yet)")

# --- File upload ---
audio_file = st.file_uploader(
    "Upload an audio file (UI test only for now)",
    type=["mp3", "wav", "m4a", "flac", "ogg"]
)

if audio_file is not None:
    st.success(f"File loaded: {audio_file.name}")
    st.audio(audio_file)
else:
    st.info("Choose a file to test the UI")

# --- Start button (placeholder flow) ---
start = st.button("Start transcription", disabled=(audio_file is None))

if start:
    if audio_file is None:
        st.warning("Please upload an audio file first.")
        st.stop()

    with st.spinner("Processing (placeholder)..."):
        time.sleep(2)  # simulate work

        # Placeholder "result" (no AI yet)
        segments = [
            {"start": 0.0, "end": 2.5, "text": "Hello and welcome."},
            {"start": 2.5, "end": 7.0, "text": "This is a placeholder transcript."},
            {"start": 7.0, "end": 11.0, "text": "The AI model is not connected yet."},
        ]
        transcript_text = "\n".join(s["text"] for s in segments)

    st.success("Done! (placeholder)")

    with st.expander("Transcript (preview)"):
        st.text_area("Text", transcript_text, height=200)

    with st.expander("Timestamps (preview)"):
        for i, s in enumerate(segments, start=1):
            st.write(f"{i:03d} [{s['start']:.2f} → {s['end']:.2f}] {s['text']}")

    st.caption("Next step: wire faster-whisper for real transcription.")
