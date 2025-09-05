# voice-notes-ai

Local-first web app that turns audio into structured, searchable notes — with transcription, speaker diarization, summaries, and exports.

## Features
- Upload common audio formats: mp3, wav, m4a, flac, ogg.
- On-device transcription with **faster-whisper** (CPU-friendly; no cloud by default).
- Speaker diarization (“who spoke when”) via **WhisperX**.
- Summaries & action items using a local LLM via **Ollama**.
- Semantic search over transcripts (embeddings + **FAISS**).
- Exports: **TXT**, **SRT** (timestamps), **JSON** (structured).
- Multi-page UI (Upload • Review • Insights) built with **Streamlit**.
- Hebrew/English support; RTL-aware rendering where relevant.
- Privacy by design: all processing stays on your machine unless you opt into cloud.

## Tech
Python 3.11+, Streamlit, faster-whisper, WhisperX, FAISS/Chroma, Ollama (optional), FFmpeg.

## Quick start
1) Create a virtual environment and install deps (see `requirements.txt`).  
2) Run: `streamlit run app.py`  
3) Open `http://localhost:8501`, upload an audio file, and explore.

## License
MIT (optional).
