# ui_style.py
import streamlit as st

BRAND = "#7c3aed"           # violet
BRAND_DARK = "#5b21b6"
ACCENT = "#22d3ee"          # cyan
GOOD = "#10b981"            # green
WARN = "#f59e0b"            # amber
BAD  = "#ef4444"            # red
MUTED = "#94a3b8"           # slate-400

def inject_css():
    st.markdown(
        f"""
<style>
:root {{
  --brand: {BRAND};
  --brand-dark: {BRAND_DARK};
  --accent: {ACCENT};
  --good: {GOOD};
  --warn: {WARN};
  --bad: {BAD};
  --muted: {MUTED};
}}

/* Layout headroom so the title icon is not clipped */
.block-container {{ padding-top: 1.6rem; }}                 /* a bit more space up top */
.block-container h1:first-of-type {{ margin-top: .15rem; }}  /* ensure first h1 isn't flush */
.stApp h1 {{ line-height: 1.25; overflow: visible; }}        /* give emoji ascenders room */
.stApp h1 span[role="img"], .stApp h1 img, .stApp h1 svg {{
  position: relative; top: .08em; display: inline-block;     /* tiny nudge down */
}}

footer {{ visibility: hidden; }}

/* Buttons */
.stButton>button, .stDownloadButton>button {{
  border-radius: 9999px;
  padding: .55rem 1rem;
  border: 1px solid rgba(255,255,255,.08);
  background: linear-gradient(135deg, var(--brand) 0%, var(--brand-dark) 100%);
  color: #fff;
  transition: transform .08s ease, box-shadow .2s ease, opacity .2s ease;
}}
.stButton>button:hover, .stDownloadButton>button:hover {{
  transform: translateY(-1px);
  box-shadow: 0 6px 18px rgba(124,58,237,.35);
  opacity: .95;
}}
button.k-secondary {{
  background: rgba(255,255,255,.06) !important;
  border: 1px solid rgba(255,255,255,.14) !important;
  color: #e5e7eb !important;
}}
.stButton>button>div[data-testid="stMarkdownContainer"] p {{ margin: 0; }}

/* Expanders */
details.st-expander {{
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,.08);
  background: rgba(255,255,255,.03);
}}
details.st-expander>summary {{ font-weight: 600; }}
details.st-expander[open] {{
  background: rgba(255,255,255,.04);
  border-color: rgba(255,255,255,.14);
}}

/* Dataframe */
div.stDataFrame, .stDataFrame [data-testid="stTable"] {{
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,.08);
}}

/* Pills */
.pills {{ display:flex; gap:.4rem; flex-wrap: wrap; align-items:center; }}
.pill {{
  display:inline-flex; align-items:center; gap:.35rem;
  border-radius: 9999px; padding:.2rem .6rem; font-size:.85rem;
  border: 1px solid rgba(255,255,255,.12);
  background: rgba(255,255,255,.06); color:#e5e7eb;
}}
.pill--brand {{ background: rgba(124,58,237,.14); border-color: rgba(124,58,237,.45); }}
.pill--good  {{ background: rgba(16,185,129,.14); border-color: rgba(16,185,129,.45); }}
.pill--warn  {{ background: rgba(245,158,11,.14); border-color: rgba(245,158,11,.45); }}
.pill--muted {{ background: rgba(148,163,184,.14); border-color: rgba(148,163,184,.45); }}

/* KPI cards */
.kpi {{
  border-radius:16px; padding:1rem 1.1rem;
  background: linear-gradient(135deg, rgba(124,58,237,.25), rgba(2,6,23,.4));
  border:1px solid rgba(255,255,255,.08);
}}
.kpi .k {{ font-size: .92rem; color:#cbd5e1; }}
.kpi .v {{ font-size: 1.6rem; font-weight:700; }}
.kpi .d {{ font-size: .9rem; color:#a3a3a3; }}
</style>
        """,
        unsafe_allow_html=True,
    )

def pill(text: str, kind: str = "muted", icon: str | None = None) -> str:
    cls = {"brand":"pill--brand","good":"pill--good","warn":"pill--warn","muted":"pill--muted"}.get(kind,"pill--muted")
    icon_html = f"<span>{icon}</span>" if icon else ""
    return f'<span class="pill {cls}">{icon_html}{text}</span>'

def kpi(label: str, value: str, delta: str | None = None) -> str:
    delta_html = f'<div class="d">{delta}</div>' if delta else ""
    return f'''
    <div class="kpi"><div class="k">{label}</div><div class="v">{value}</div>{delta_html}</div>
    '''
