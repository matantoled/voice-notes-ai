import os, glob
from typing import Tuple
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_DIR = "outputs"

st.set_page_config(page_title="Semantic search", layout="wide")
st.title("🔎 Semantic search")

def s_to_timestamp(s: float) -> str:
    h = int(s // 3600); m = int((s % 3600) // 60); sec = int(s % 60); ms = int((s - int(s)) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)

def load_all_npz() -> Tuple[np.ndarray, list, list, list, list]:
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.npz")))
    if not files:
        return None, [], [], [], []
    embs, starts, ends, texts, bases = [], [], [], [], []
    for path in files:
        data = np.load(path, allow_pickle=True)
        emb = data["emb"].astype(np.float32)
        stts = data["starts"].astype(np.float32)
        ends_ = data["ends"].astype(np.float32)
        txts = list(data["texts"])
        base = os.path.splitext(os.path.basename(path))[0]
        embs.append(emb)
        starts.append(stts)
        ends.append(ends_)
        texts.extend(txts)
        bases.extend([base] * len(txts))
    emb_all = np.vstack(embs) if embs else None
    starts = np.concatenate(starts) if starts else np.array([], dtype=np.float32)
    ends   = np.concatenate(ends)   if ends   else np.array([], dtype=np.float32)
    return emb_all, list(starts), list(ends), texts, bases

def outputs_signature() -> str:
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.npz")))
    parts = [f"{os.path.basename(p)}:{int(os.path.getmtime(p))}" for p in files]
    return "|".join(parts)

@st.cache_resource
def build_index(sig: str):
    emb_all, starts, ends, texts, bases = load_all_npz()
    if emb_all is None or len(texts) == 0:
        return None, None, None, None, None, None
    d = emb_all.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine (embeddings are normalized)
    index.add(emb_all)
    return index, emb_all, starts, ends, list(texts), bases

if not os.path.isdir(OUTPUT_DIR):
    st.info("No outputs yet. Transcribe something first in the main page.")
    st.stop()

# ---- Sidebar controls ----
with st.sidebar:
    top_k = st.slider("Top-K results", min_value=3, max_value=20, value=5, step=1)
    min_score = st.slider("Min similarity", min_value=0.0, max_value=1.0, value=0.10, step=0.01)
    hide_dups = st.checkbox("Hide near-duplicates", value=True)
    if st.button("Refresh index"):
        try:
            st.cache_resource.clear()
        finally:
            st.rerun()

sig = outputs_signature()
index, emb_all, starts, ends, texts, bases = build_index(sig)
if index is None:
    st.info("No embeddings found yet. Run a new transcription so a .npz is saved.")
    st.stop()

# Filter by source (which runs to include)
unique_sources = sorted(set(bases))
with st.sidebar:
    chosen_sources = st.multiselect("Filter sources", unique_sources, default=unique_sources)

q = st.text_input("Ask or search (semantic)", "")
if not q:
    st.caption("Type a query to search across all saved transcripts.")
    st.stop()

with st.spinner("Searching..."):
    embedder = get_embedder()
    qv = embedder.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(qv.astype(np.float32), top_k * 5)  # fetch more to allow filtering/dedup

def norm_text(t: str) -> str:
    return " ".join(t.lower().split())

seen = set()
kept = []
for idx, score in zip(I[0], D[0]):
    if idx < 0:
        continue
    base = bases[idx]
    if base not in chosen_sources:
        continue
    if score < min_score:
        continue
    text = texts[idx]
    if hide_dups:
        key = norm_text(text)
        if key in seen:
            continue
        seen.add(key)
    kept.append({
        "score": float(score),
        "start": float(starts[idx]),
        "end": float(ends[idx]),
        "text": text,
        "base": base,
    })
    if len(kept) >= top_k:
        break

st.subheader("Results")
if not kept:
    st.info("No results with current filters. Try lowering Min similarity or disabling duplicate hiding.")
else:
    for r in kept:
        with st.expander(f"[{s_to_timestamp(r['start'])} → {s_to_timestamp(r['end'])}] • {r['base']} • score={r['score']:.3f}"):
            # Copy-friendly blocks
            st.caption("Timestamp")
            st.code(f"{s_to_timestamp(r['start'])} --> {s_to_timestamp(r['end'])}")
            st.caption("Text")
            st.code(r["text"])

            # Download single-cue SRT for the snippet
            snippet_srt = f"1\n{s_to_timestamp(r['start'])} --> {s_to_timestamp(r['end'])}\n{r['text'].strip()}\n"
            st.download_button("⬇ SRT (this snippet)", snippet_srt.encode("utf-8"),
                               file_name=f"{r['base']}_{int(r['start'])}-{int(r['end'])}.srt",
                               key=f"dl_snip_{r['base']}_{r['start']:.2f}")
