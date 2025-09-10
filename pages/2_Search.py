import os, glob, json, base64, mimetypes, re
from typing import Tuple, List
from ui_style import inject_css, pill, kpi
inject_css()

import numpy as np
import faiss
import streamlit as st
import streamlit.components.v1 as components
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_DIR = "outputs"

st.set_page_config(page_title="Semantic search", layout="wide")
st.title("🔎 Semantic search")

# ---------- optional TextRank (sumy) ----------
_SUMY_OK = True
try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer
except Exception:
    _SUMY_OK = False

# ---------- utils ----------
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

# ---------- transcript/summary helpers ----------
def _read_txt(base: str) -> tuple[str, float] | tuple[None, None]:
    p = os.path.join(OUTPUT_DIR, f"{base}.txt")
    if not os.path.exists(p):
        return None, None
    try:
        with open(p, encoding="utf-8", errors="ignore") as f:
            return f.read(), os.path.getmtime(p)
    except Exception:
        return None, None

def _read_json_text(base: str) -> tuple[str, float] | tuple[None, None]:
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
    txt, mt = _read_txt(base)
    if txt is not None:
        return txt, mt
    return _read_json_text(base)

def _summarize_textrank(text: str, sentences: int = 6) -> str:
    if not text.strip():
        return ""
    if _SUMY_OK:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, sentences)
        return "\n".join(str(s) for s in summary).strip()
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:sentences]).strip()

def _extract_action_items(text: str, limit: int = 10) -> list[str]:
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

# ---------- audio helpers ----------
AUDIO_EXTS = [".mp3", ".wav", ".m4a", ".flac", ".ogg"]

def meta_json_path(base: str) -> str:
    return os.path.join(OUTPUT_DIR, f"{base}.json")

def find_audio_for_base(base: str) -> str | None:
    """Find audio file for given transcript base (via JSON meta or by scanning)."""
    # 1) Try json meta 'audio_saved'
    jp = meta_json_path(base)
    if os.path.exists(jp):
        try:
            with open(jp, encoding="utf-8") as f:
                meta = json.load(f)
            aud_rel = meta.get("audio_saved")
            if aud_rel:
                p = os.path.join(OUTPUT_DIR, aud_rel)
                if os.path.exists(p):
                    return p
        except Exception:
            pass
    # 2) Fallback: scan by known extensions
    for ext in AUDIO_EXTS:
        p = os.path.join(OUTPUT_DIR, f"{base}{ext}")
        if os.path.exists(p):
            return p
    return None

@st.cache_resource
def load_audio_b64(path: str) -> tuple[str, str]:
    """Return (mime, base64) for the audio file so we can embed in HTML."""
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "audio/mpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return mime, b64

def snippet_player_html(mime: str, b64: str, start: float, end: float, key: str) -> str:
    """Small HTML audio with 'Play snippet' button that seeks to [start,end]."""
    safe_key = key.replace(".", "_").replace(":", "_")
    html_doc = f"""
    <div id="wrap_{safe_key}" style="font-family: ui-sans-serif,system-ui; margin: 6px 0;">
      <audio id="aud_{safe_key}" controls preload="metadata" style="width:100%;">
        <source src="data:{mime};base64,{b64}">
      </audio>
      <div style="margin-top:6px;">
        <button id="btn_{safe_key}" style="padding:4px 10px; border:1px solid #888; border-radius:6px; cursor:pointer;">
          ▶ Play snippet ({start:.1f}s → {end:.1f}s)
        </button>
        <button id="pause_{safe_key}" style="padding:4px 10px; margin-left:6px; border:1px solid #888; border-radius:6px; cursor:pointer;">
          ⏸ Pause
        </button>
      </div>
      <script>
        (function() {{
          const a = document.getElementById("aud_{safe_key}");
          const btn = document.getElementById("btn_{safe_key}");
          const pp = document.getElementById("pause_{safe_key}");
          const start = {start:.3f};
          const end = {end:.3f};
          let guard = false;
          btn.addEventListener("click", () => {{
            if (!a) return;
            a.currentTime = start;
            a.play();
            guard = true;
          }});
          pp.addEventListener("click", () => a && a.pause());
          a.addEventListener("timeupdate", () => {{
            if (!guard) return;
            if (a.currentTime >= end) {{
              a.pause();
              guard = false;
            }}
          }});
        }})();
      </script>
    </div>
    """
    return html_doc

# ---------- page preconditions ----------
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
kept: List[dict] = []
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

            # Snippet SRT download
            snippet_srt = f"1\n{s_to_timestamp(r['start'])} --> {s_to_timestamp(r['end'])}\n{r['text'].strip()}\n"
            st.download_button("⬇ SRT (this snippet)",
                               snippet_srt.encode("utf-8"),
                               file_name=f"{r['base']}_{int(r['start'])}-{int(r['end'])}.srt",
                               key=f"dl_snip_{r['base']}_{r['start']:.2f}")

            # Inline audio snippet player (if audio exists)
            audio_path = find_audio_for_base(r["base"])
            if audio_path and os.path.exists(audio_path):
                mime, b64 = load_audio_b64(audio_path)
                html_player = snippet_player_html(mime, b64, r["start"], r["end"], key=f"{r['base']}_{r['start']:.2f}")
                components.html(html_player, height=120, scrolling=False)
            else:
                st.warning("Audio not found for this transcript. To enable snippet playback, keep the original audio under outputs/ (add 'audio_saved' to JSON or place a file named like the base).")

            # -------- Summary & Action items for this base --------
            st.markdown("---")
            st.markdown("**Summary & action items**")
            _, src_mtime = transcript_text_for_base(r["base"])
            mtime_key = int(src_mtime or 0)
            with st.spinner("Loading..."):
                summary, actions = get_summary_for_base(r["base"], mtime_key)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Summary**")
                st.write(summary or "(no summary)")
                st.download_button(
                    "⬇ Summary (TXT)",
                    (summary or "(no summary)").encode("utf-8"),
                    file_name=f"{r['base']}.summary.txt",
                    key=f"sum_dl_{r['base']}_{r['start']:.2f}",
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
                    file_name=f"{r['base']}.action.txt",
                    key=f"act_dl_{r['base']}_{r['start']:.2f}",
                )
