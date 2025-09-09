import os, glob, json
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
    text_preview = "\n".join(s.get("text", "") for s in meta.get("segments", []))

    if q and q.lower() not in text_preview.lower():
        continue

    shown += 1
    with st.expander(f"{base} — {meta.get('language','?').upper()} — {meta.get('source_file','')}"):
        st.caption(
            f"Saved: {meta.get('saved_at')} • Model: {meta.get('model')} "
            f"({meta.get('compute_type')}) • VAD: {meta.get('vad')}"
        )
        # >>> fix: unique key per item
        st.text_area("Preview", text_preview[:4000], height=200, key=f"preview_{base}")

        col1, col2, col3 = st.columns([1, 1, 1])
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
