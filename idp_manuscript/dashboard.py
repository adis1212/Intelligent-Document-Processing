"""
IDP Manuscript Processing System — Streamlit Dashboard
========================================================
Full pipeline with session-state-driven UI and manuscript theme.

Architecture:
    - All pipeline results stored in st.session_state.results
    - Display code renders from stored state (survives reruns)
    - Gallery navigation uses st.session_state.selected_page_idx
    - Search uses st.session_state.search_query

Changes from original:
    - FIXED: Page navigation no longer resets pipeline state
    - REDESIGNED: Parchment/manuscript theme, serif fonts, brown accents
    - ADDED: Skew status badges with threshold display
    - ADDED: Smart mock OCR (always shows valid text)
    - ADDED: Metadata-driven search bar in aggregation section
    - ADDED: Summary cards + download buttons in output section
    - ADDED: Score breakdown (Quality × 40% + OCR × 40% + Sequence × 20%)
"""

import os, sys, time
import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config_loader import load_config
from utils.sample_generator import generate_sample_manuscripts
from modules.preprocessor import load_batch, preprocess_image
from modules.quality_checker import run_quality_check, check_skew
from modules.sequence_detector import run_sequence_detection
from modules.ocr_engine import run_ocr
from modules.aggregator import (
    aggregate_results, generate_batch_summary,
    build_search_index, search_documents,
)
from modules.output_generator import generate_csv, generate_json_audit

# ─── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="IDP Manuscript Processing System",
    page_icon="📜", layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Clean Dashboard Theme ──────────────────────────────────────
# White background, black text, colorful numbers, clear readability
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=DM+Sans:wght@400;500;600;700&display=swap');

/* ── Global ─────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', 'DM Sans', -apple-system, sans-serif;
    color: #1a1a1a;
}
.stApp {
    background: #ffffff;
}
/* Force all Streamlit text elements to black */
.stApp p, .stApp span, .stApp label, .stApp div,
.stApp .stMarkdown, .stApp li, .stApp td, .stApp th {
    color: #1a1a1a !important;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    color: #111111 !important;
}
/* Streamlit metric values and labels */
[data-testid="stMetricValue"] {
    color: #1a1a1a !important;
}
[data-testid="stMetricLabel"] {
    color: #444444 !important;
}
/* Captions */
.stApp .stCaption, .stApp small {
    color: #666666 !important;
}

/* ── Sidebar ────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    border-right: 3px solid #3b82f6;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    color: #ffffff !important;
    border: 1px solid #60a5fa;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.2s ease;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(59,130,246,0.4);
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #059669, #10b981);
    border: 2px solid #34d399;
}

/* ── Main header ────────────────────────────── */
.main-header {
    background: linear-gradient(135deg, #1e293b, #334155, #1e293b);
    padding: 2rem 2.5rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    text-align: center;
    border: 2px solid #3b82f6;
    position: relative;
}
.main-header h1 {
    color: #ffffff !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 2.2rem;
    margin: 0;
    letter-spacing: 0.5px;
    font-weight: 700;
    position: relative;
}
.main-header p {
    color: #93c5fd !important;
    font-size: 1.05rem;
    margin: 0.4rem 0 0;
    font-style: italic;
    position: relative;
}

/* ── Section pills ──────────────────────────── */
.section-pill {
    display: inline-block;
    background: linear-gradient(90deg, #1e293b, #334155);
    color: #ffffff !important;
    padding: 0.45rem 1.2rem;
    border-radius: 20px;
    font-size: 1.05rem;
    font-weight: 600;
    margin: 1.2rem 0 0.6rem 0;
    border-left: 4px solid #3b82f6;
    letter-spacing: 0.5px;
    font-family: 'DM Sans', sans-serif;
}
.pill-done  { border-left-color: #22c55e; }
.pill-run   { border-left-color: #f97316; }
.pill-error { border-left-color: #ef4444; }

/* ── Algorithm cards ────────────────────────── */
.algo-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.algo-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1e293b !important;
    margin-bottom: 0.3rem;
    font-family: 'DM Sans', sans-serif;
}
.algo-desc {
    font-size: 0.85rem;
    color: #475569 !important;
    line-height: 1.55;
}

/* ── Image labels ───────────────────────────── */
.img-label {
    text-align: center;
    background: #1e293b;
    border-radius: 8px;
    padding: 4px 0;
    font-size: 0.8rem;
    color: #ffffff !important;
    margin-bottom: 4px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
}

/* ── Status badges ──────────────────────────── */
.status-ready  { color: #16a34a !important; font-weight: 700; }
.status-review { color: #ea580c !important; font-weight: 700; }
.status-reject { color: #dc2626 !important; font-weight: 700; }
.status-ok     { color: #16a34a !important; }
.status-warn   { color: #ea580c !important; }

/* ── Skew badge ─────────────────────────────── */
.skew-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 8px;
    font-size: 0.9rem;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
}
.skew-ok   { background: #dcfce7; color: #15803d !important; border: 1px solid #86efac; }
.skew-fail { background: #fef2f2; color: #b91c1c !important; border: 1px solid #fca5a5; }

/* ── Summary cards ──────────────────────────── */
.summary-card {
    background: #ffffff;
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.4rem 1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: transform 0.2s, box-shadow 0.2s;
}
.summary-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
}
.summary-card .sc-value {
    font-size: 2.4rem;
    font-weight: 800;
    font-family: 'DM Sans', sans-serif;
}
.summary-card .sc-label {
    font-size: 0.85rem;
    color: #64748b !important;
    margin-top: 0.3rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
/* Color-coded summary card values */
.sc-total   .sc-value { color: #2563eb !important; }
.sc-ready   .sc-value { color: #16a34a !important; }
.sc-review  .sc-value { color: #ea580c !important; }
.sc-reject  .sc-value { color: #dc2626 !important; }
.sc-rate    .sc-value { color: #7c3aed !important; }
/* Colored top borders for cards */
.sc-total   { border-top: 4px solid #3b82f6; }
.sc-ready   { border-top: 4px solid #22c55e; }
.sc-review  { border-top: 4px solid #f97316; }
.sc-reject  { border-top: 4px solid #ef4444; }
.sc-rate    { border-top: 4px solid #8b5cf6; }

/* ── Search result card ─────────────────────── */
.search-result {
    background: #f8fafc;
    border-left: 4px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.search-result .sr-title {
    font-weight: 700; color: #1e293b !important; font-size: 0.95rem;
}
.search-result .sr-snippet {
    color: #475569 !important; font-size: 0.88rem; margin-top: 0.3rem;
    line-height: 1.5; font-style: italic;
}

/* ── Duplicate badge ────────────────────────── */
.dup-badge {
    background: #7c3aed; color: #fff !important; border-radius: 6px;
    font-size: 0.72rem; padding: 2px 8px; margin-left: 6px;
}

/* ── Score breakdown ────────────────────────── */
.score-breakdown {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.9rem;
    color: #1e293b !important;
}

/* ── Dataframe styling ──────────────────────── */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
}

/* ── Expander headers black text ────────────── */
.streamlit-expanderHeader p, .streamlit-expanderHeader span {
    color: #1a1a1a !important;
    font-weight: 600;
}

/* ── Streamlit chrome — kept visible for navigation ── */
</style>
""", unsafe_allow_html=True)


# ─── Helpers ────────────────────────────────────────────────────
def to_pil(img):
    """Convert OpenCV image (BGR or gray) to PIL for Streamlit display."""
    if img is None: return None
    if len(img.shape) == 2: return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def pill(text, kind=""):
    """Render a styled section pill."""
    css = f"section-pill {'pill-'+kind if kind else ''}"
    st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)

def status_badge(text):
    """Return HTML span with color-coded status."""
    t = text.upper()
    if "READY" in t or " OK" in t or "PASS" in t:
        return f'<span class="status-ready">{text}</span>'
    if "REVIEW" in t or "SKEWED" in t or "CROPPED" in t or "OCCLUDED" in t:
        return f'<span class="status-review">{text}</span>'
    return f'<span class="status-reject">{text}</span>'

def skew_badge_html(angle, threshold, is_skewed):
    """Render a skew status badge: 'Skew: 7.2° (Threshold: 5°) → ❌ SKEWED'"""
    if is_skewed:
        return (f'<span class="skew-badge skew-fail">'
                f'Skew: {angle}° (Threshold: {threshold}°) → ❌ SKEWED</span>')
    else:
        return (f'<span class="skew-badge skew-ok">'
                f'Skew: {angle}° (Threshold: {threshold}°) → ✅ OK</span>')

def summary_card(value, label, emoji="", css_class=""):
    """Render a styled summary card with optional color class."""
    return f'''<div class="summary-card {css_class}">
        <div class="sc-value">{emoji} {value}</div>
        <div class="sc-label">{label}</div>
    </div>'''

def projection_chart(scores, angles):
    """Mini matplotlib chart of projection profile scores by angle."""
    fig, ax = plt.subplots(figsize=(4, 2))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    ax.plot(angles, scores, color="#3b82f6", linewidth=1.5)
    best_i = int(np.argmax(scores))
    ax.axvline(angles[best_i], color="#22c55e", linestyle="--", linewidth=1.2,
               label=f"Best: {angles[best_i]:.1f}°")
    ax.set_xlabel("Angle (°)", fontsize=7, color="#1a1a1a")
    ax.set_ylabel("Variance", fontsize=7, color="#1a1a1a")
    ax.tick_params(colors="#1a1a1a", labelsize=6)
    for spine in ax.spines.values():
        spine.set_color("#e2e8f0")
    ax.legend(fontsize=6, labelcolor="#1a1a1a", facecolor="#ffffff",
              edgecolor="#e2e8f0")
    fig.tight_layout(pad=0.4)
    return fig


# ─── Session State Initialization ────────────────────────────────
# All pipeline results and UI state are stored here so they survive
# Streamlit reruns (widget clicks, gallery navigation, etc.)
def _init():
    defaults = {
        "pipe_status": {k: "pending" for k in [
            "preprocessing", "quality_check", "sequence_detection",
            "ocr_processing", "aggregation", "output_generation"
        ]},
        "results": None,          # Full pipeline results dict
        "running": False,         # Pipeline currently executing
        "selected_page_idx": 0,   # Gallery: which page is selected
        "search_query": "",       # Search bar value
        "search_results": [],     # Cached search results
        "uploaded_files_saved": False,  # Whether uploads have been saved
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ─── Header ──────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>📜 Manuscript Processing Dashboard</h1>
  <p>Intelligent Document Processing · Heritage Manuscript Archives</p>
</div>
""", unsafe_allow_html=True)


# ─── Upload directory ────────────────────────────────────────────
_UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "uploads")
os.makedirs(_UPLOAD_DIR, exist_ok=True)

_SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "sample_images")


def _save_uploaded_files(uploaded_files):
    """Save uploaded files to working directory and return the path."""
    # Clear previous uploads
    for f in os.listdir(_UPLOAD_DIR):
        fp = os.path.join(_UPLOAD_DIR, f)
        if os.path.isfile(fp):
            os.remove(fp)

    saved = []
    for uf in uploaded_files:
        dest = os.path.join(_UPLOAD_DIR, uf.name)
        with open(dest, "wb") as out:
            out.write(uf.getbuffer())
        saved.append(uf.name)
    return saved


# ─── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.divider()

    # ── File uploader — primary way to add images ──────────────
    st.markdown("### 📁 Upload Manuscript Images")
    uploaded_files = st.file_uploader(
        "Select your manuscript images",
        type=["png", "jpg", "jpeg", "tiff", "tif", "bmp"],
        accept_multiple_files=True,
        help="Browse and select one or more manuscript images from your computer."
    )

    # Save uploaded files to disk when new files are received
    if uploaded_files:
        saved_names = _save_uploaded_files(uploaded_files)
        folder_path = _UPLOAD_DIR
        st.success(f"✅ {len(saved_names)} image(s) uploaded")
        for fn in saved_names:
            st.caption(f"  📄 {fn}")
    else:
        folder_path = _UPLOAD_DIR  # default to uploads dir

    # ── Small fallback: generate sample images for quick testing
    with st.expander("🎨 Or use sample images"):
        if st.button("Generate 6 Sample Manuscripts", use_container_width=True):
            with st.spinner("Generating samples..."):
                files = generate_sample_manuscripts(_SAMPLE_DIR, 6)
                # Copy samples to upload dir so pipeline can find them
                import shutil
                for f in os.listdir(_UPLOAD_DIR):
                    fp = os.path.join(_UPLOAD_DIR, f)
                    if os.path.isfile(fp):
                        os.remove(fp)
                for f in files:
                    shutil.copy2(f, _UPLOAD_DIR)
                st.success(f"✅ {len(files)} sample images ready")
                st.rerun()

    st.divider()

    run_btn = st.button("🚀 Run Full Pipeline", use_container_width=True,
                        type="primary", disabled=st.session_state.running)

    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.pipe_status = {k: "pending"
                                        for k in st.session_state.pipe_status}
        st.session_state.results = None
        st.session_state.running = False
        st.session_state.selected_page_idx = 0
        st.session_state.search_query = ""
        st.session_state.search_results = []
        st.session_state.uploaded_files_saved = False
        st.rerun()

    st.divider()
    st.markdown("### 📊 Pipeline Progress")

    icons = {"pending": "⬜", "running": "🔄", "completed": "✅", "error": "❌"}
    labels = {
        "preprocessing":      "1 · Preprocessing",
        "quality_check":      "2 · Quality Check",
        "sequence_detection": "3 · Sequence Detection",
        "ocr_processing":     "4 · OCR Processing",
        "aggregation":        "5 · Aggregation",
        "output_generation":  "6 · Output Generation",
    }
    for k, label in labels.items():
        icon = icons.get(st.session_state.pipe_status.get(k, "pending"), "⬜")
        st.markdown(f"{icon} {label}")

    # ── Config display (read-only) ────────────────────────────
    st.divider()
    with st.expander("📋 Configuration"):
        try:
            config_display = load_config()
            for section, values in config_display.items():
                st.markdown(f"**{section}**")
                if isinstance(values, dict):
                    for k, v in values.items():
                        st.caption(f"  {k}: `{v}`")
                else:
                    st.caption(f"  {values}")
        except Exception:
            st.caption("Could not load config.")


# ═══════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ═══════════════════════════════════════════════════════════════
# This block only runs when the user clicks "Run Full Pipeline".
# All results are stored in st.session_state.results so the
# display code below can render them on every rerun.
# ═══════════════════════════════════════════════════════════════
if run_btn:
    # Check that images exist in the uploads directory
    has_images = os.path.isdir(folder_path) and any(
        f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'))
        for f in os.listdir(folder_path)
    )
    if not has_images:
        st.error("❌ No images found!")
        st.info("👆 Please **upload your manuscript images** using the file picker in the sidebar, then click **Run Full Pipeline**.")
        st.stop()

    st.session_state.running = True
    config = load_config()
    R = {
        "preprocessing": [], "quality": [], "sequence": None,
        "ocr": [], "aggregated": [], "batch_summary": {},
        "search_index": None, "csv_path": None, "json_path": None,
        "config": config,
    }

    pbar = st.progress(0, text="Starting pipeline…")

    # ── STEP 1 — PREPROCESSING ────────────────────────────────
    st.session_state.pipe_status["preprocessing"] = "running"
    pbar.progress(5, text="Loading & preprocessing images…")

    try:
        imgs = load_batch(folder_path,
                          config["preprocessing"]["supported_formats"])
        if not imgs:
            st.error("No supported images found!")
            st.session_state.running = False
            st.stop()

        for img_data in imgs:
            p = preprocess_image(img_data["image"], config["preprocessing"])
            p["filename"] = img_data["filename"]
            p["path"]     = img_data["path"]
            R["preprocessing"].append(p)

        st.session_state.pipe_status["preprocessing"] = "completed"
        pbar.progress(16, text="Preprocessing done ✅")

    except Exception as e:
        st.session_state.pipe_status["preprocessing"] = "error"
        st.error(f"❌ Preprocessing failed: {e}")
        st.session_state.running = False
        st.stop()

    # ── STEP 2 — QUALITY CHECK ────────────────────────────────
    st.session_state.pipe_status["quality_check"] = "running"
    pbar.progress(20, text="Running quality checks…")

    try:
        qconf = config["quality"]
        for prep in R["preprocessing"]:
            img = prep.get("processed", prep.get("original"))
            R["quality"].append(run_quality_check(img, qconf))

        st.session_state.pipe_status["quality_check"] = "completed"
        pbar.progress(38, text="Quality check done ✅")

    except Exception as e:
        st.session_state.pipe_status["quality_check"] = "error"
        st.error(f"❌ Quality check failed: {e}")

    # ── STEP 3 — SEQUENCE DETECTION ───────────────────────────
    st.session_state.pipe_status["sequence_detection"] = "running"
    pbar.progress(42, text="Detecting sequences & duplicates…")

    try:
        seqconf = config["sequence"]
        fnames = [p["filename"] for p in R["preprocessing"]]
        images_data = [{"filename": p["filename"],
                        "image": p.get("processed", p.get("original"))}
                       for p in R["preprocessing"]]
        R["sequence"] = run_sequence_detection(fnames, seqconf,
                                               images_data=images_data)

        st.session_state.pipe_status["sequence_detection"] = "completed"
        pbar.progress(55, text="Sequence detection done ✅")

    except Exception as e:
        st.session_state.pipe_status["sequence_detection"] = "error"
        st.error(f"❌ Sequence detection failed: {e}")
        import traceback; st.code(traceback.format_exc())

    # ── STEP 4 — OCR ─────────────────────────────────────────
    st.session_state.pipe_status["ocr_processing"] = "running"
    pbar.progress(60, text="Running OCR…")

    try:
        ocrconf = config["ocr"]
        for prep in R["preprocessing"]:
            img = prep.get("processed", prep.get("original"))
            R["ocr"].append(run_ocr(img, ocrconf))

        st.session_state.pipe_status["ocr_processing"] = "completed"
        pbar.progress(75, text="OCR done ✅")

    except Exception as e:
        st.session_state.pipe_status["ocr_processing"] = "error"
        st.error(f"❌ OCR failed: {e}")

    # ── STEP 5 — AGGREGATION ─────────────────────────────────
    st.session_state.pipe_status["aggregation"] = "running"
    pbar.progress(82, text="Aggregating results…")

    try:
        R["aggregated"] = aggregate_results(
            R["preprocessing"], R["quality"], R["sequence"], R["ocr"],
            config["aggregation"])
        R["batch_summary"] = generate_batch_summary(R["aggregated"])

        # Build search index
        R["search_index"] = build_search_index(
            R["ocr"], R["preprocessing"], R["quality"], R["aggregated"])

        st.session_state.pipe_status["aggregation"] = "completed"
        pbar.progress(92, text="Aggregation done ✅")

    except Exception as e:
        st.session_state.pipe_status["aggregation"] = "error"
        st.error(f"❌ Aggregation failed: {e}")

    # ── STEP 6 — OUTPUT ──────────────────────────────────────
    st.session_state.pipe_status["output_generation"] = "running"
    pbar.progress(96, text="Generating reports…")

    try:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               config["output"]["output_dir"])
        os.makedirs(out_dir, exist_ok=True)
        csv_path  = os.path.join(out_dir, config["output"]["csv_filename"])
        json_path = os.path.join(out_dir, config["output"]["json_filename"])
        R["csv_path"]  = generate_csv(R["aggregated"], csv_path)
        R["json_path"] = generate_json_audit(R["aggregated"],
                                              R["batch_summary"], json_path)

        st.session_state.pipe_status["output_generation"] = "completed"
        pbar.progress(100, text="Pipeline complete ✅")

    except Exception as e:
        st.session_state.pipe_status["output_generation"] = "error"
        st.error(f"❌ Output failed: {e}")

    # ── Store everything and rerun to display ─────────────────
    st.session_state.results = R
    st.session_state.running = False
    st.session_state.selected_page_idx = 0
    st.balloons()
    st.rerun()


# ═══════════════════════════════════════════════════════════════
# DISPLAY SECTION — Renders from st.session_state.results
# ═══════════════════════════════════════════════════════════════
# This code runs on EVERY Streamlit rerun. It reads from stored
# session state, so navigating the gallery, searching, etc.
# never causes data loss.
# ═══════════════════════════════════════════════════════════════

R = st.session_state.results

if R is not None:
    n = len(R["preprocessing"])
    fnames = [p["filename"] for p in R["preprocessing"]]

    # ══════════════════════════════════════════════════════════
    # IMAGE GALLERY — Clickable grid of all manuscript pages
    # ══════════════════════════════════════════════════════════
    pill("🖼️ Manuscript Gallery", "done")

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📷 Total Pages", n)
    m2.metric("⚙️ Pipeline Steps", 6)
    bs = R.get("batch_summary", {})
    m3.metric("✅ Ready", bs.get("ready", "—"))
    m4.metric("📈 Pass Rate", f'{bs.get("pass_rate", "—")}%')

    # Gallery grid (3 per row) — select by clicking
    st.markdown("**Select a page to inspect:**")
    rows = (n + 2) // 3
    for row_i in range(rows):
        cols = st.columns(3)
        for ci, col in enumerate(cols):
            idx = row_i * 3 + ci
            if idx >= n:
                break
            prep = R["preprocessing"][idx]
            with col:
                # Highlight selected
                border = "3px solid #3b82f6" if idx == st.session_state.selected_page_idx else "1px solid #e2e8f0"
                st.markdown(
                    f'<div style="border:{border};border-radius:10px;'
                    f'padding:4px;margin-bottom:6px;background:#ffffff;">'
                    f'<div class="img-label">📄 {prep["filename"]}</div></div>',
                    unsafe_allow_html=True)
                st.image(to_pil(prep.get("processed", prep.get("original"))),
                         use_container_width=True)
                if st.button(f"Select", key=f"sel_{idx}",
                             use_container_width=True):
                    st.session_state.selected_page_idx = idx
                    st.rerun()

    # ── Selected Page Preview ─────────────────────────────────
    st.markdown("---")
    si = st.session_state.selected_page_idx
    sp = R["preprocessing"][si]

    pill(f"🔎 Inspecting: {sp['filename']}")

    bc1, bc2 = st.columns(2)
    with bc1:
        st.markdown("**Original**")
        st.image(to_pil(sp["original"]), use_container_width=True)
        oh, ow = sp["original"].shape[:2]
        st.caption(f"{ow}×{oh} px")
    with bc2:
        st.markdown("**After Preprocessing**")
        st.image(to_pil(sp["processed"]), use_container_width=True)
        ph, pw = sp["processed"].shape[:2]
        st.caption(f"{pw}×{ph} px · 4 steps applied")

    # Step-by-step view for selected page
    with st.expander(f"⚙️ Preprocessing Steps: {sp['filename']}"):
        sc1, sc2, sc3, sc4 = st.columns(4)
        step_map = [("grayscale", "① Grayscale", sc1),
                    ("resized", "② Resize", sc2),
                    ("denoised", "③ Denoise", sc3),
                    ("enhanced", "④ CLAHE", sc4)]
        for skey, slabel, scol in step_map:
            if skey in sp["steps"]:
                with scol:
                    st.caption(slabel)
                    st.image(to_pil(sp["steps"][skey]),
                             use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # STEP 1 — PREPROCESSING DETAILS
    # ══════════════════════════════════════════════════════════
    with st.expander("📋 Step 1 · Preprocessing Details"):
        with st.container():
            st.markdown("""
            <div class="algo-card"><div class="algo-title">Pipeline: Grayscale → Resize → NLM Denoise → CLAHE</div>
            <div class="algo-desc">
            <b>①</b> BGR→Gray reduces 3 channels to 1.
            <b>②</b> INTER_AREA resize normalises to consistent working size.
            <b>③</b> Non-Local Means removes scan noise while preserving edges.
            <b>④</b> CLAHE boosts local contrast in dark manuscript areas.
            </div></div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # STEP 2 — QUALITY CHECK
    # ══════════════════════════════════════════════════════════
    with st.expander("🔬 Step 2 · Quality Check", expanded=True):

        # Algorithm explanation
        qa1, qa2, qa3, qa4 = st.columns(4)
        with qa1:
            st.markdown('<div class="algo-card"><div class="algo-title">Blur · Laplacian</div>'
                        '<div class="algo-desc">Laplacian edge detector variance. '
                        'High variance = sharp. Threshold: 100.</div></div>',
                        unsafe_allow_html=True)
        with qa2:
            st.markdown('<div class="algo-card"><div class="algo-title">Skew · PPM</div>'
                        '<div class="algo-desc">Projection Profile Method. Rotates '
                        '−20°→+20°, finds max row-sum variance.</div></div>',
                        unsafe_allow_html=True)
        with qa3:
            st.markdown('<div class="algo-card"><div class="algo-title">Crop · Margins</div>'
                        '<div class="algo-desc">Checks 2% border strips for content. '
                        'Content at every edge = over-cropped.</div></div>',
                        unsafe_allow_html=True)
        with qa4:
            st.markdown('<div class="algo-card"><div class="algo-title">Occlusion · Dark%</div>'
                        '<div class="algo-desc">Counts pixels &lt;30 intensity. '
                        '&gt;15% dark = occlusion detected.</div></div>',
                        unsafe_allow_html=True)

        # Results table
        qdata = []
        for prep, qr in zip(R["preprocessing"], R["quality"]):
            qdata.append({
                "Filename": prep["filename"],
                "Blur Score": qr["blur"]["blur_score"],
                "Blur": qr["blur"]["blur_status"],
                "Skew (°)": qr["skew"]["skew_angle"],
                "Skew": qr["skew"]["skew_status"],
                "Crop": qr["crop"]["crop_status"],
                "Occlusion %": qr["occlusion"]["occlusion_percent"],
                "Occlusion": qr["occlusion"]["occlusion_status"],
                "Quality Score": qr["quality_score"],
                "Status": qr["overall_status"],
            })
        df_q = pd.DataFrame(qdata)
        st.dataframe(df_q, use_container_width=True, hide_index=True)

        # Quality metrics
        avg_q  = round(sum(q["quality_score"] for q in R["quality"]) / len(R["quality"]), 1)
        blurred = sum(1 for q in R["quality"] if q["blur"]["is_blurred"])
        skewed  = sum(1 for q in R["quality"] if q["skew"]["is_skewed"])
        qm1, qm2, qm3, qm4 = st.columns(4)
        qm1.metric("⭐ Avg Quality", f"{avg_q}/100")
        qm2.metric("🔍 Blurred", blurred,
                    delta=f"-{blurred}" if blurred else None,
                    delta_color="inverse")
        qm3.metric("📐 Skewed", skewed,
                    delta=f"-{skewed}" if skewed else None,
                    delta_color="inverse")
        qm4.metric("✅ Quality Pass",
                    sum(1 for q in R["quality"]
                        if q["overall_status"] == "PASS"))

        # ── Skew deep-dive for selected page ──────────────────
        st.markdown("**📐 Skew Detection · Selected Page**")
        qr_sel = R["quality"][si]

        # Skew badge
        skew_angle = qr_sel["skew"]["skew_angle"]
        skew_thresh = qr_sel["skew"].get("threshold_used", 5.0)
        is_skewed = qr_sel["skew"]["is_skewed"]
        st.markdown(skew_badge_html(skew_angle, skew_thresh, is_skewed),
                    unsafe_allow_html=True)

        skc1, skc2, skc3 = st.columns([1, 1, 1])
        with skc1:
            st.markdown("**Input Image**")
            st.image(to_pil(sp["processed"]), use_container_width=True)
        with skc2:
            st.markdown("**Deskewed Output**")
            corrected = qr_sel["skew"].get("corrected_image")
            if corrected is not None:
                st.image(to_pil(corrected), use_container_width=True)
            else:
                st.info("No correction needed")
        with skc3:
            st.markdown("**Projection Profile**")
            angles  = qr_sel["skew"].get("angles_tested", [])
            scores  = qr_sel["skew"].get("projection_scores", [])
            if angles and scores:
                fig = projection_chart(scores, angles)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            st.metric("Detected Angle", f'{skew_angle}°',
                      delta="Straight" if not is_skewed
                            else f"Skewed {abs(skew_angle):.1f}°",
                      delta_color="normal" if not is_skewed else "inverse")

    # ══════════════════════════════════════════════════════════
    # STEP 3 — SEQUENCE DETECTION
    # ══════════════════════════════════════════════════════════
    with st.expander("🔢 Step 3 · Sequence Detection + Duplicates"):

        # Algorithm explanation
        sa1, sa2, sa3 = st.columns(3)
        with sa1:
            st.markdown('<div class="algo-card"><div class="algo-title">Page Detection · Regex</div>'
                        '<div class="algo-desc">Extracts page numbers from filenames. '
                        'Primary: page_(\\d+). Fallback: first number.</div></div>',
                        unsafe_allow_html=True)
        with sa2:
            st.markdown('<div class="algo-card"><div class="algo-title">Gap Detection</div>'
                        '<div class="algo-desc">Builds expected range(min,max+1). '
                        'Missing = expected − actual. Flags incomplete batches.</div></div>',
                        unsafe_allow_html=True)
        with sa3:
            st.markdown('<div class="algo-card"><div class="algo-title">Duplicates · dHash</div>'
                        '<div class="algo-desc">64-bit perceptual hash per image. '
                        'Hamming distance ≤ 10 → duplicate. Works across resizes.</div></div>',
                        unsafe_allow_html=True)

        # Page number table
        if R["sequence"]:
            dup_flags = {}
            if R["sequence"]["duplicate_analysis"]:
                dup_flags = R["sequence"]["duplicate_analysis"].get("dup_flags", {})

            seq_rows = []
            for sr in R["sequence"]["per_file"]:
                is_dup = sr.get("is_duplicate", False)
                seq_rows.append({
                    "Filename": sr["filename"],
                    "Page #": sr["page_number"] if sr["page_number"] is not None else "—",
                    "Method": sr["method"],
                    "Duplicate?": "⚠️ YES" if is_dup else "✅ NO",
                    "Status": sr["status"],
                })
            st.dataframe(pd.DataFrame(seq_rows),
                         use_container_width=True, hide_index=True)

            # Gap analysis
            gap = R["sequence"]["gap_analysis"]
            sg1, sg2, sg3, sg4 = st.columns(4)
            sg1.metric("📄 Pages Found", gap["total_detected"])
            sg2.metric("📚 Expected", gap.get("expected_count", "—"))
            sg3.metric("❓ Missing", len(gap["missing_pages"]))
            sg4.metric("✅ Complete?",
                       "YES" if gap.get("is_complete") else "NO")

            if gap["missing_pages"]:
                st.warning(f"⚠️ Missing pages: {gap['missing_pages']}")
            if gap["duplicates"]:
                st.warning(f"⚠️ Duplicate page numbers: {gap['duplicates']}")

            # Duplicate content
            dup_an = R["sequence"]["duplicate_analysis"]
            if dup_an:
                st.markdown("**🔁 Duplicate Image Detection (dHash)**")
                if dup_an["total_duplicates"] == 0:
                    st.success("✅ No duplicate images detected.")
                else:
                    st.warning(f"⚠️ {dup_an['total_duplicates']} duplicate pair(s) found!")
                    for dp in dup_an["duplicates"]:
                        fn1, fn2 = dp["pair"]
                        st.markdown(
                            f"- **{fn1}** ↔ **{fn2}** "
                            f"(Hamming: `{dp['hamming_distance']}`, "
                            f"Similarity: `{dp['similarity_percent']}%`)")

                with st.expander("🔑 Image Hashes"):
                    hash_df = pd.DataFrame(
                        [{"Filename": fn, "dHash": hv}
                         for fn, hv in dup_an["hashes"].items()])
                    st.dataframe(hash_df, use_container_width=True,
                                 hide_index=True)

    # ══════════════════════════════════════════════════════════
    # STEP 4 — OCR
    # ══════════════════════════════════════════════════════════
    with st.expander("📝 Step 4 · OCR Processing"):

        st.markdown('<div class="algo-card"><div class="algo-title">'
                    'OCR Engine · Text Extraction</div>'
                    '<div class="algo-desc">Extracts English text from '
                    'manuscript images using optical character recognition. '
                    'Confidence scores are derived from image quality analysis. '
                    'Keywords are extracted using word frequency analysis. '
                    'Script detection identifies the writing system via '
                    'Unicode range classification.'
                    '</div></div>', unsafe_allow_html=True)

        for i, (prep, ocr_r) in enumerate(zip(R["preprocessing"], R["ocr"])):
            with st.expander(
                f"📄 {prep['filename']} — Confidence: {ocr_r['confidence']}%",
                expanded=(i == si)):
                oc1, oc2 = st.columns([2, 1])
                with oc1:
                    st.markdown("**Extracted Text:**")
                    txt = ocr_r["text"][:600] + ("…" if len(ocr_r["text"]) > 600 else "")
                    st.text_area("", txt, height=120,
                                 key=f"ocr_{i}", disabled=True)
                with oc2:
                    st.metric("🎯 Confidence", f'{ocr_r["confidence"]}%')
                    st.metric("📊 Words", ocr_r["word_count"])
                    st.metric("🔤 Script", ocr_r["script"]["script"])
                    if ocr_r["keywords"]:
                        st.markdown("**Top Keywords:**")
                        for w, c in ocr_r["keywords"][:5]:
                            st.markdown(f"- `{w}` ({c}×)")

    # ══════════════════════════════════════════════════════════
    # STEP 5 — AGGREGATION + SEARCH
    # ══════════════════════════════════════════════════════════
    with st.expander("📊 Step 5 · Aggregation & Search", expanded=True):

        # ── Summary cards ─────────────────────────────────────
        bs = R.get("batch_summary", {})
        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        with sc1:
            st.markdown(summary_card(bs.get("total_images", 0),
                                     "Total Pages", "📷", "sc-total"),
                        unsafe_allow_html=True)
        with sc2:
            st.markdown(summary_card(bs.get("ready", 0),
                                     "Ready", "✅", "sc-ready"),
                        unsafe_allow_html=True)
        with sc3:
            st.markdown(summary_card(bs.get("review", 0),
                                     "Review", "⚠️", "sc-review"),
                        unsafe_allow_html=True)
        with sc4:
            st.markdown(summary_card(bs.get("rejected", 0),
                                     "Rejected", "❌", "sc-reject"),
                        unsafe_allow_html=True)
        with sc5:
            st.markdown(summary_card(f'{bs.get("pass_rate", 0)}%',
                                     "Pass Rate", "📈", "sc-rate"),
                        unsafe_allow_html=True)

        st.markdown("")

        # ── Score breakdown table ─────────────────────────────
        st.markdown("**📋 Score Breakdown**")
        st.caption("Final Score = Quality (40%) + OCR Confidence (40%) + Sequence (20%)")

        breakdown_data = []
        for agg in R["aggregated"]:
            breakdown_data.append({
                "Filename": agg["filename"],
                "Quality (40%)": agg.get("quality_component", "—"),
                "OCR (40%)": agg.get("ocr_component", "—"),
                "Sequence (20%)": agg.get("sequence_component", "—"),
                "Final Score": agg["final_score"],
                "Status": agg["status"],
                "Blur": agg["blur_status"],
                "Skew": agg["skew_status"],
            })
        st.dataframe(pd.DataFrame(breakdown_data),
                     use_container_width=True, hide_index=True)

        # ── Document Search ───────────────────────────────────
        st.markdown("---")
        st.markdown("**🔍 Document Search**")
        st.caption("Search across extracted text, filenames, and keywords to locate specific content.")

        search_q = st.text_input(
            "Search manuscripts…",
            value=st.session_state.search_query,
            placeholder="e.g., India, manuscript, calligraphy",
            key="search_input")

        if search_q != st.session_state.search_query:
            st.session_state.search_query = search_q

        if search_q and R.get("search_index"):
            results = search_documents(search_q, R["search_index"])
            st.session_state.search_results = results

            if results:
                st.success(f"Found **{len(results)}** matching page(s)")
                for sr in results:
                    pg = sr["page_number"]
                    fn = sr["filename"]
                    status = sr["status"]
                    kw_tag = " 🏷️ Keyword match" if sr["keyword_match"] else ""

                    st.markdown(
                        f'<div class="search-result">'
                        f'<div class="sr-title">Page {pg} · {fn} '
                        f'({status}){kw_tag}</div>',
                        unsafe_allow_html=True)
                    for snippet in sr["snippets"][:2]:
                        # Bold the query in the snippet
                        highlighted = snippet.replace(
                            search_q,
                            f"**{search_q}**")
                        st.markdown(
                            f'<div class="sr-snippet">"{highlighted}"</div>',
                            unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info(f"No results found for '{search_q}'")
        elif search_q and not R.get("search_index"):
            st.warning("Search index not available. Re-run the pipeline.")

    # ══════════════════════════════════════════════════════════
    # STEP 6 — OUTPUT
    # ══════════════════════════════════════════════════════════
    with st.expander("💾 Step 6 · Output & Reports", expanded=True):

        # ── Summary cards ─────────────────────────────────────
        bs = R.get("batch_summary", {})
        oc1, oc2, oc3, oc4, oc5 = st.columns(5)
        with oc1:
            st.markdown(summary_card(bs.get("total_images", 0),
                                     "Total Pages", "📷", "sc-total"),
                        unsafe_allow_html=True)
        with oc2:
            st.markdown(summary_card(bs.get("ready", 0),
                                     "Ready", "✅", "sc-ready"),
                        unsafe_allow_html=True)
        with oc3:
            st.markdown(summary_card(bs.get("review", 0),
                                     "Need Review", "⚠️", "sc-review"),
                        unsafe_allow_html=True)
        with oc4:
            st.markdown(summary_card(bs.get("rejected", 0),
                                     "Rejected", "❌", "sc-reject"),
                        unsafe_allow_html=True)
        with oc5:
            st.markdown(summary_card(f'{bs.get("pass_rate", 0)}%',
                                     "Pass Rate", "📈", "sc-rate"),
                        unsafe_allow_html=True)

        st.markdown("")

        # ── Final results table ───────────────────────────────
        st.markdown("**📋 Final Results Table**")
        final_df = pd.DataFrame(R["aggregated"])
        # Rename for clean display
        display_cols = {
            "filename": "Filename",
            "quality_score": "Quality",
            "page_number": "Page #",
            "ocr_confidence": "OCR Conf.",
            "quality_component": "Q (40%)",
            "ocr_component": "OCR (40%)",
            "sequence_component": "Seq (20%)",
            "final_score": "Final Score",
            "status": "Status",
            "blur_status": "Blur",
            "skew_status": "Skew",
        }
        final_display = final_df.rename(columns=display_cols)
        # Select only the columns we want to show
        show_cols = [c for c in display_cols.values() if c in final_display.columns]
        st.dataframe(final_display[show_cols],
                     use_container_width=True, hide_index=True)

        # ── Download buttons ──────────────────────────────────
        st.markdown("---")
        st.markdown("**📥 Download Reports**")

        dc1, dc2 = st.columns(2)
        if R.get("csv_path") and os.path.exists(R["csv_path"]):
            with open(R["csv_path"], "r", encoding="utf-8") as f:
                csv_data = f.read()
            dc1.download_button(
                "📥 Download CSV Report",
                data=csv_data,
                file_name="processing_results.csv",
                mime="text/csv",
                use_container_width=True)

        if R.get("json_path") and os.path.exists(R["json_path"]):
            with open(R["json_path"], "r", encoding="utf-8") as f:
                json_data = f.read()
            dc2.download_button(
                "📥 Download JSON Audit Log",
                data=json_data,
                file_name="audit_log.json",
                mime="application/json",
                use_container_width=True)

    # ── Pipeline complete banner ──────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:1.5rem;
    background:linear-gradient(135deg,#1e293b,#334155);
    border-radius:14px;border:2px solid #3b82f6;margin-top:1rem;">
      <h2 style="color:#ffffff;margin:0;font-family:'DM Sans',sans-serif;">
      📜 Pipeline Completed Successfully</h2>
      <p style="color:#93c5fd;margin-top:0.4rem;font-style:italic;">
      All 6 modules finished · Download your reports above</p>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# IDLE STATE — shown when no pipeline has been run yet
# ═══════════════════════════════════════════════════════════════
if R is None and not st.session_state.running:
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
### 🚀 Quick Start
1. **Upload** your manuscript images using the sidebar
2. Click **Run Full Pipeline**
3. Browse the manuscript gallery
4. Inspect quality, skew, OCR, and duplicates
5. Search across documents
6. Download CSV + JSON reports
        """)
    with c2:
        st.markdown("""
### 📋 Modules & Algorithms
| # | Module | Algorithm |
|---|--------|-----------|
| 1 | Preprocessing | Grayscale · Resize · NLM · CLAHE |
| 2 | Quality Check | Laplacian · **PPM Skew** · Margin · Dark% |
| 3 | Sequence + Dups | Regex · **dHash** · Hamming distance |
| 4 | OCR | Text Extraction · Confidence · Script detect |
| 5 | Aggregation | Weighted scoring · **Document Search** |
| 6 | Output | CSV + JSON audit log |
        """)
    st.info("👆 **Upload your manuscript images** in the sidebar, then click **Run Full Pipeline** to start.")
