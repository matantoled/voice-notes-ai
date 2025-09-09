import os, glob, json
import streamlit as st

st.set_page_config(page_title="Recent transcripts", layout="wide")
st.title("🗂️ Recent transcripts")

if not os.path.isdir("outputs"):
    st.info("No outputs yet. Run a transcription first.")
    st.stop()

q = st.text_input("Search (simple contains match)", "")
paths = sorted(glob.glob("outputs/*.json"), reverse=True)

for path in paths:
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)

    base = os.path.splitext(os.path.basename(path))[0]
    text_preview = "\n".join(s.get("text", "") for s in meta.get("segments", []))

    if q and q.lower() not in text_preview.lower():
        continue

    with st.expander(f"{base} — {meta.get('language','?').upper()} — {meta.get('source_file','')}"):
        st.caption(
            f"Saved: {meta.get('saved_at')} • Model: {meta.get('model')} "
            f"({meta.get('compute_type')}) • VAD: {meta.get('vad')}"
        )
        st.text_area("Preview", text_preview[:4000], height=200)

        col1, col2 = st.columns(2)
        with col1:
            txt_path = f"outputs/{base}.txt"
            if os.path.exists(txt_path):
                with open(txt_path, "rb") as f:
                    st.download_button("⬇ TXT", f.read(), file_name=f"{base}.txt")
        with col2:
            srt_path = f"outputs/{base}.srt"
            if os.path.exists(srt_path):
                with open(srt_path, "rb") as f:
                    st.download_button("⬇ SRT", f.read(), file_name=f"{base}.srt")
