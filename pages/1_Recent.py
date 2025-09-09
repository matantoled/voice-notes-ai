import os, glob, json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Recent transcripts", layout="wide")
st.title("🗂️ Recent transcripts")

OUTPUT_DIR = "outputs"

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
    for ext in (".npz", ".mp3", ".wav", ".m4a", ".flac", ".ogg"):
        p = os.path.join(OUTPUT_DIR, f"{base}{ext}")
        if os.path.exists(p):
            safe_remove(p)

if not os.path.isdir(OUTPUT_DIR):
    st.info("No outputs yet. Run a transcription first.")
    st.stop()

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
            df = pd.DataFrame({
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

if shown == 0:
    st.info("No results match your search.")
