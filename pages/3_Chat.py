# pages/3_Chat.py
import os, glob, json, base64, mimetypes, re
from typing import Tuple, List, Dict
from collections import defaultdict
from ui_style import inject_css, pill, kpi
inject_css()


import numpy as np
import faiss
import streamlit as st
import streamlit.components.v1 as components
from sentence_transformers import SentenceTransformer

# ---------- optional summarizer (sumy) ----------
_SUMY_OK = True
try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer
except Exception:
    _SUMY_OK = False

# ---------- optional NER (spaCy multilingual); has graceful fallback ----------
_SPACY_OK = True
try:
    import spacy
    try:
        _NLP = spacy.load("xx_ent_wiki_sm")   # multilingual PERSON
    except Exception:
        _SPACY_OK = False
except Exception:
    _SPACY_OK = False

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_DIR = "outputs"

st.set_page_config(page_title="Transcript Q&A", layout="wide")
st.title("💬 Transcript Q&A")
st.caption("Ask anything about your transcripts. I'll search and summarize locally.")

# ============== utils (shared with search page) ==============
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
        embs.append(emb); starts.append(stts); ends.append(ends_)
        texts.extend(txts); bases.extend([base]*len(txts))
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

# ---------- audio helpers ----------
AUDIO_EXTS = [".mp3", ".wav", ".m4a", ".flac", ".ogg"]

def meta_json_path(base: str) -> str:
    return os.path.join(OUTPUT_DIR, f"{base}.json")

def find_audio_for_base(base: str) -> str | None:
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
    for ext in AUDIO_EXTS:
        p = os.path.join(OUTPUT_DIR, f"{base}{ext}")
        if os.path.exists(p):
            return p
    return None

@st.cache_resource
def load_audio_b64(path: str) -> tuple[str, str]:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "audio/mpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return mime, b64

def snippet_player_html(mime: str, b64: str, start: float, end: float, key: str) -> str:
    safe_key = key.replace(".", "_").replace(":", "_")
    return f"""
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
        (function(){{
          const a = document.getElementById("aud_{safe_key}");
          const btn = document.getElementById("btn_{safe_key}");
          const pp = document.getElementById("pause_{safe_key}");
          const start = {start:.3f}, end = {end:.3f};
          let guard = false;
          btn.addEventListener("click", () => {{ if(!a) return; a.currentTime = start; a.play(); guard = true; }});
          pp.addEventListener("click", () => a && a.pause());
          a.addEventListener("timeupdate", () => {{ if(guard && a.currentTime >= end) {{ a.pause(); guard = false; }} }});
        }})();
      </script>
    </div>
    """

# ---------- summarization / NER helpers ----------
def summarize_text(text: str, sentences: int = 3) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if _SUMY_OK:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summ = TextRankSummarizer()(parser.document, sentences)
        out = " ".join(str(s) for s in summ).strip()
        if out:
            return out
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:sentences]).strip()

_CAP_STOP = set("""
I I'm I've I'd I'll You You're You'll He She's We We're They'll It It's They
Monday Tuesday Wednesday Thursday Friday Saturday Sunday
Hi Hello Okay Bye Thanks Thank You Yeah Yes No
""".split())

_CAP_RX = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")

def extract_persons(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if _SPACY_OK:
        doc = _NLP(text)
        names = {ent.text.strip() for ent in doc.ents if ent.label_ == "PER"}
    else:
        # naive: capitalized sequences (English/Euro languages)
        names = set()
        for m in _CAP_RX.finditer(text):
            cand = m.group(1).strip()
            if cand.split()[0] in _CAP_STOP:
                continue
            if len(cand) < 2:
                continue
            names.add(cand)
    # small cleanup
    cleaned = sorted({re.sub(r"\s+", " ", n) for n in names if len(n) <= 60})
    return cleaned[:50]

# ============== page preconditions ==============
if not os.path.isdir(OUTPUT_DIR):
    st.info("No outputs yet. Transcribe something first in the main page.")
    st.stop()

# ---- sidebar ----
with st.sidebar:
    st.header("Options")
    top_k = st.slider("Top-K segments", 3, 25, 8, 1)
    min_score = st.slider("Min similarity", 0.0, 1.0, 0.30, 0.01)
    show_players = st.checkbox("Show audio snippet players", value=True)
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

# filter by source
unique_sources = sorted(set(bases))
with st.sidebar:
    chosen_sources = st.multiselect("Limit to sources", unique_sources, default=unique_sources)

# ============== query box ==============
q = st.text_input("Type your question…", placeholder="e.g., List all participant names across the recordings")
if not q:
    st.stop()

# ============== retrieve ==============
with st.spinner("Searching…"):
    embedder = get_embedder()
    qv = embedder.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(qv.astype(np.float32), top_k * 6)  # fetch more, we'll filter/dedup

def norm_text(t: str) -> str:
    return " ".join(t.lower().split())

seen = set()
hits: List[dict] = []
for idx, score in zip(I[0], D[0]):
    if idx < 0:
        continue
    base = bases[idx]
    if base not in chosen_sources or score < min_score:
        continue
    text = texts[idx]
    key = norm_text(text)
    if key in seen:
        continue
    seen.add(key)
    hits.append({
        "score": float(score),
        "start": float(starts[idx]),
        "end": float(ends[idx]),
        "text": text,
        "base": base,
    })
    if len(hits) >= top_k:
        break

# ============== compose answer ==============
def compose_answer(query: str, results: List[dict]) -> str:
    if not results:
        return "No relevant segments were found with the current filters."

    # special case: user asks for names/participants
    if re.search(r"\b(name|names|participants?)\b", query.lower()):
        bucket = defaultdict(list)
        for r in results:
            bucket[r["base"]].append(r["text"])
        all_text = "\n".join(t for arr in bucket.values() for t in arr)
        people = extract_persons(all_text)
        if not people:
            return "No person names were detected in the retrieved segments."
        lines = ["**Detected participant names:**"] + [f"- {p}" for p in people]
        return "\n".join(lines)

    # default: brief per-recording summaries + a key quote
    by_base: Dict[str, List[dict]] = defaultdict(list)
    for r in results:
        by_base[r["base"]].append(r)

    out = ["**Answer (auto-composed from top matches):**"]
    for base, arr in by_base.items():
        arr = sorted(arr, key=lambda x: -x["score"])
        joined = " ".join(x["text"] for x in arr)
        summary = summarize_text(joined, sentences=2) or "(no summary)"
        top = arr[0]
        quote = top["text"].strip().replace("\n", " ")
        out.append(f"\n- **{base}**")
        out.append(f"  - Summary: {summary}")
        out.append(f"  - Key quote [{s_to_timestamp(top['start'])} → {s_to_timestamp(top['end'])}]: “{quote}”")
    return "\n".join(out)

answer_md = compose_answer(q, hits)
st.markdown(answer_md)
st.download_button("⬇ Copy answer (TXT)", answer_md.encode("utf-8"), "answer.txt")

st.markdown("### Sources")
if not hits:
    st.info("No results with current filters. Try raising similarity or changing sources.")
else:
    for r in hits:
        with st.expander(f"{r['base']} | {s_to_timestamp(r['start'])} → {s_to_timestamp(r['end'])} | score={r['score']:.3f}"):
            st.code(r["text"])
            if show_players:
                audio_path = find_audio_for_base(r["base"])
                if audio_path and os.path.exists(audio_path):
                    mime, b64 = load_audio_b64(audio_path)
                    html_player = snippet_player_html(mime, b64, r["start"], r["end"], key=f"{r['base']}_{r['start']:.2f}")
                    components.html(html_player, height=120, scrolling=False)
