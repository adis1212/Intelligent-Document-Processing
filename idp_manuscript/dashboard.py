"""
IDP Manuscript Processing System — Streamlit Dashboard
========================================================
Full pipeline with all algorithms live in the UI.
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
from modules.aggregator import aggregate_results, generate_batch_summary
from modules.output_generator import generate_csv, generate_json_audit

# ─── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="IDP Manuscript Processing System",
    page_icon="📜", layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    padding: 2rem 2.5rem; border-radius: 18px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 40px rgba(0,0,0,0.4);
    text-align: center; border: 1px solid #3d3a6b;
}
.main-header h1 { color: #e8d5b7; font-size: 2.4rem; margin:0; letter-spacing:1.5px; font-weight:700; }
.main-header p  { color: #a0aec0; font-size: 1rem; margin: 0.4rem 0 0; }

.section-pill {
    display:inline-block; background:linear-gradient(90deg,#302b63,#24243e);
    color:#e8d5b7; padding:0.45rem 1.2rem; border-radius:20px;
    font-size:1.05rem; font-weight:600; margin:1.2rem 0 0.6rem 0;
    border-left:4px solid #c9a84c; letter-spacing:0.5px;
}
.pill-done   { border-left-color: #00c853; }
.pill-run    { border-left-color: #ffa000; }
.pill-error  { border-left-color: #ff1744; }

.algo-card {
    background: #0e1117; border:1px solid #2d3148;
    border-radius:12px; padding:1rem 1.2rem; margin-bottom:0.8rem;
}
.algo-title { font-size:0.95rem; font-weight:600; color:#c9a84c; margin-bottom:0.3rem; }
.algo-desc  { font-size:0.82rem; color:#8892a6; line-height:1.5; }

.img-label {
    text-align:center; background:#16213e; border-radius:8px;
    padding:3px 0; font-size:0.78rem; color:#e8d5b7; margin-bottom:4px;
}
.status-ready  { color:#00c853; font-weight:700; }
.status-review { color:#ffa000; font-weight:700; }
.status-reject { color:#ff1744; font-weight:700; }
.status-ok     { color:#00c853; }
.status-warn   { color:#ffa000; }

.dup-badge { background:#7b1fa2; color:#fff; border-radius:6px;
             font-size:0.72rem; padding:2px 8px; margin-left:6px; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ────────────────────────────────────────────────────
def to_pil(img):
    if img is None: return None
    if len(img.shape) == 2: return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def pill(text, kind=""):
    css = f"section-pill {'pill-'+kind if kind else ''}"
    st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)

def status_badge(text):
    t = text.upper()
    if "READY" in t or " OK" in t or "PASS" in t:
        return f'<span class="status-ready">{text}</span>'
    if "REVIEW" in t or "SKEWED" in t or "CROPPED" in t or "OCCLUDED" in t:
        return f'<span class="status-review">{text}</span>'
    return f'<span class="status-reject">{text}</span>'

def projection_chart(scores, angles):
    """Mini matplotlib chart of projection profile scores by angle."""
    fig, ax = plt.subplots(figsize=(4, 2))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.plot(angles, scores, color="#c9a84c", linewidth=1.5)
    best_i = int(np.argmax(scores))
    ax.axvline(angles[best_i], color="#00c853", linestyle="--", linewidth=1.2,
               label=f"Best: {angles[best_i]:.1f}°")
    ax.set_xlabel("Angle (°)", fontsize=7, color="#8892a6")
    ax.set_ylabel("Variance", fontsize=7, color="#8892a6")
    ax.tick_params(colors="#8892a6", labelsize=6)
    ax.spines[:].set_color("#2d3148")
    ax.legend(fontsize=6, labelcolor="#e8d5b7", facecolor="#0e1117")
    fig.tight_layout(pad=0.4)
    return fig


# ─── Session state ───────────────────────────────────────────────
def _init():
    defaults = {
        "pipe_status": {k: "pending" for k in ["preprocessing","quality_check",
                         "sequence_detection","ocr_processing","aggregation","output_generation"]},
        "results": None,
        "running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()


# ─── Header ──────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>📜 IDP Manuscript Processing System</h1>
  <p>Intelligent Document Processing · Heritage Manuscript Archives</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.divider()

    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_images")
    folder_path = st.text_input("📁 Image Folder", value=default_dir)

    if st.button("🎨 Generate Sample Images", use_container_width=True):
        with st.spinner("Generating..."):
            files = generate_sample_manuscripts(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_images"), 6)
            st.success(f"✅ {len(files)} images created")

    st.divider()

    run_btn = st.button("🚀 Run Full Pipeline", use_container_width=True, type="primary",
                        disabled=st.session_state.running)

    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.pipe_status = {k: "pending" for k in st.session_state.pipe_status}
        st.session_state.results = None
        st.session_state.running = False
        st.rerun()

    st.divider()
    st.markdown("### 📊 Pipeline Progress")

    icons = {"pending": "⬜", "running": "🔄", "completed": "✅", "error": "❌"}
    labels = {
        "preprocessing": "1. Preprocessing",
        "quality_check": "2. Quality Check",
        "sequence_detection": "3. Sequence Detection",
        "ocr_processing": "4. OCR Processing",
        "aggregation": "5. Aggregation",
        "output_generation": "6. Output",
    }
    for k, label in labels.items():
        icon = icons.get(st.session_state.pipe_status.get(k, "pending"), "⬜")
        st.markdown(f"{icon} {label}")


# ═══════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ═══════════════════════════════════════════════════════════════
if run_btn:
    if not os.path.isdir(folder_path):
        st.error(f"❌ Folder not found: `{folder_path}`")
        st.info("Click **Generate Sample Images** in the sidebar first.")
        st.stop()

    st.session_state.running = True
    config = load_config()
    R = {"preprocessing":[], "quality":[], "sequence":None, "ocr":[], "aggregated":[], "batch_summary":{}}

    pbar = st.progress(0, text="Starting pipeline…")

    # ══════════════════════════════════════════════════════════
    # STEP 1 — PREPROCESSING
    # ══════════════════════════════════════════════════════════
    pill("📋 Step 1 · Preprocessing", "run")
    st.session_state.pipe_status["preprocessing"] = "running"

    with st.spinner("Loading & preprocessing all images…"):
        try:
            imgs = load_batch(folder_path, config["preprocessing"]["supported_formats"])
            if not imgs:
                st.error("No supported images found!"); st.stop()

            m1, m2, m3 = st.columns(3)
            m1.metric("📷 Images Loaded", len(imgs))
            m2.metric("📁 Folder", os.path.basename(folder_path))
            m3.metric("⚙️ Pipeline Steps", 4)

            for img_data in imgs:
                p = preprocess_image(img_data["image"], config["preprocessing"])
                p["filename"] = img_data["filename"]
                p["path"]     = img_data["path"]
                R["preprocessing"].append(p)

            n = len(R["preprocessing"])

            # ── Algorithm explanation card ─────────────────────
            with st.expander("ℹ️ Preprocessing Algorithm Details"):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown('<div class="algo-card"><div class="algo-title">① Grayscale</div>'
                                '<div class="algo-desc">Convert BGR→Gray. Reduces 3 channels to 1, halving '
                                'data while preserving texture and edge information needed for analysis.</div></div>',
                                unsafe_allow_html=True)
                with c2:
                    st.markdown('<div class="algo-card"><div class="algo-title">② Resize</div>'
                                '<div class="algo-desc">Scale to target resolution using INTER_AREA '
                                '(best for downscaling). Normalises images to a consistent working size.</div></div>',
                                unsafe_allow_html=True)
                with c3:
                    st.markdown('<div class="algo-card"><div class="algo-title">③ Denoise (NLM)</div>'
                                '<div class="algo-desc">Non-Local Means removes scan noise by averaging '
                                'similar patches across the image — preserves edges unlike simple blur.</div></div>',
                                unsafe_allow_html=True)
                with c4:
                    st.markdown('<div class="algo-card"><div class="algo-title">④ CLAHE</div>'
                                '<div class="algo-desc">Contrast Limited Adaptive Histogram Equalisation '
                                'boosts local contrast in dark manuscript areas without over-exposing bright regions.</div></div>',
                                unsafe_allow_html=True)

            # ── ALL IMAGES GRID (3 per row) ────────────────────
            st.markdown("**🖼️ All Pages — Before vs After**")
            rows = (n + 2) // 3
            for row_i in range(rows):
                cols = st.columns(3)
                for ci, col in enumerate(cols):
                    idx = row_i * 3 + ci
                    if idx >= n:
                        break
                    prep = R["preprocessing"][idx]
                    with col:
                        st.markdown(f'<div class="img-label">📄 {prep["filename"]}</div>',
                                    unsafe_allow_html=True)
                        t1, t2 = st.tabs(["Original", "Processed"])
                        with t1:
                            st.image(to_pil(prep["original"]), use_container_width=True)
                        with t2:
                            st.image(to_pil(prep["processed"]), use_container_width=True)

            # ── Manual Page Browser ────────────────────────────
            st.markdown("---")
            st.markdown("**🔎 Manual Page Browser — select any page to inspect**")
            fnames = [p["filename"] for p in R["preprocessing"]]
            sel = st.selectbox("Select page:", fnames, key="prep_sel")
            si = fnames.index(sel)
            sp = R["preprocessing"][si]

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

            # Pipeline steps for selected page
            with st.expander(f"⚙️ Step-by-step: {sel}"):
                sc1, sc2, sc3, sc4 = st.columns(4)
                step_map = [("grayscale","1·Gray",sc1),("resized","2·Resize",sc2),
                            ("denoised","3·Denoise",sc3),("enhanced","4·CLAHE",sc4)]
                for skey, slabel, scol in step_map:
                    if skey in sp["steps"]:
                        with scol:
                            st.caption(slabel)
                            st.image(to_pil(sp["steps"][skey]), use_container_width=True)

            st.session_state.pipe_status["preprocessing"] = "completed"
            pill(f"✅ Preprocessing Done · {n} images", "done")
            pbar.progress(16, text="Preprocessing done")
            time.sleep(0.2)

        except Exception as e:
            st.session_state.pipe_status["preprocessing"] = "error"
            st.error(f"❌ Preprocessing failed: {e}")
            st.session_state.running = False; st.stop()

    # ══════════════════════════════════════════════════════════
    # STEP 2 — QUALITY CHECK
    # ══════════════════════════════════════════════════════════
    pill("🔬 Step 2 · Quality Check", "run")
    st.session_state.pipe_status["quality_check"] = "running"
    pbar.progress(20, text="Running quality checks…")

    with st.spinner("Analysing image quality…"):
        try:
            qconf = config["quality"]
            for prep in R["preprocessing"]:
                img = prep.get("processed", prep.get("original"))
                R["quality"].append(run_quality_check(img, qconf))

            # ── Algorithm explanation ──────────────────────────
            with st.expander("ℹ️ Quality Algorithms Explained"):
                qa1, qa2, qa3, qa4 = st.columns(4)
                with qa1:
                    st.markdown('<div class="algo-card"><div class="algo-title">Blur · Laplacian Variance</div>'
                                '<div class="algo-desc">Applies Laplacian edge detector. Sharp images have high '
                                'variance (many edges). Blurry images → low variance. Threshold: 100.</div></div>',
                                unsafe_allow_html=True)
                with qa2:
                    st.markdown('<div class="algo-card"><div class="algo-title">Skew · Projection Profile</div>'
                                '<div class="algo-desc">Rotates binary image at angles −20°→+20°. Measures '
                                'row-sum variance. Max variance = best alignment. Detects tilt > 3°.</div></div>',
                                unsafe_allow_html=True)
                with qa3:
                    st.markdown('<div class="algo-card"><div class="algo-title">Crop · Margin Check</div>'
                                '<div class="algo-desc">Checks 2% border strips for content pixels. '
                                'If content touches every edge → image is over-cropped.</div></div>',
                                unsafe_allow_html=True)
                with qa4:
                    st.markdown('<div class="algo-card"><div class="algo-title">Occlusion · Dark Pixel %</div>'
                                '<div class="algo-desc">Counts pixels with intensity &lt;30. If &gt;15% of '
                                'image is very dark → occlusion (thumb/stamp/damage).</div></div>',
                                unsafe_allow_html=True)

            # ── Results table ──────────────────────────────────
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

            avg_q  = round(sum(q["quality_score"] for q in R["quality"]) / len(R["quality"]), 1)
            blurred = sum(1 for q in R["quality"] if q["blur"]["is_blurred"])
            skewed  = sum(1 for q in R["quality"] if q["skew"]["is_skewed"])
            qm1, qm2, qm3, qm4 = st.columns(4)
            qm1.metric("⭐ Avg Quality", f"{avg_q}/100")
            qm2.metric("🔍 Blurred", blurred, delta=f"-{blurred}" if blurred else None, delta_color="inverse")
            qm3.metric("📐 Skewed", skewed, delta=f"-{skewed}" if skewed else None, delta_color="inverse")
            qm4.metric("✅ Quality Pass", sum(1 for q in R["quality"] if q["overall_status"]=="PASS"))

            # ── Per-image skew deep-dive ───────────────────────
            st.markdown("**📐 Skew Detection · Projection Profile Viewer**")
            sel_q = st.selectbox("Select image for skew analysis:", fnames, key="skew_sel")
            qi = fnames.index(sel_q)
            qr_sel = R["quality"][qi]
            prep_sel = R["preprocessing"][qi]

            skc1, skc2, skc3 = st.columns([1,1,1])
            with skc1:
                st.markdown("**Input Image**")
                st.image(to_pil(prep_sel["processed"]), use_container_width=True)
            with skc2:
                st.markdown("**Deskewed Output**")
                corrected = qr_sel["skew"].get("corrected_image")
                if corrected is not None:
                    st.image(to_pil(corrected), use_container_width=True)
                else:
                    st.info("No correction needed")
            with skc3:
                st.markdown("**Projection Profile Score**")
                angles  = qr_sel["skew"].get("angles_tested", [])
                scores  = qr_sel["skew"].get("projection_scores", [])
                if angles and scores:
                    fig = projection_chart(scores, angles)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                skew_val = qr_sel["skew"]["skew_angle"]
                skew_ok  = not qr_sel["skew"]["is_skewed"]
                st.metric("Detected Angle", f'{skew_val}°',
                          delta="Straight" if skew_ok else f"Skewed {abs(skew_val):.1f}°",
                          delta_color="normal" if skew_ok else "inverse")

            st.session_state.pipe_status["quality_check"] = "completed"
            pill("✅ Quality Check Done", "done")
            pbar.progress(38, text="Quality check done")
            time.sleep(0.2)

        except Exception as e:
            st.session_state.pipe_status["quality_check"] = "error"
            st.error(f"❌ Quality check failed: {e}")

    # ══════════════════════════════════════════════════════════
    # STEP 3 — SEQUENCE DETECTION (Page numbers + Duplicates)
    # ══════════════════════════════════════════════════════════
    pill("🔢 Step 3 · Sequence Detection + Duplicate Check", "run")
    st.session_state.pipe_status["sequence_detection"] = "running"
    pbar.progress(42, text="Detecting sequences & duplicates…")

    with st.spinner("Detecting page sequences and duplicate images…"):
        try:
            seqconf = config["sequence"]
            # Pass actual image data for content-based duplicate detection
            images_data = [{"filename": p["filename"],
                            "image": p.get("processed", p.get("original"))}
                           for p in R["preprocessing"]]
            R["sequence"] = run_sequence_detection(fnames, seqconf, images_data=images_data)

            # ── Algorithm explanation ──────────────────────────
            with st.expander("ℹ️ Sequence & Duplicate Algorithms Explained"):
                sa1, sa2, sa3 = st.columns(3)
                with sa1:
                    st.markdown('<div class="algo-card"><div class="algo-title">Page Detection · Regex</div>'
                                '<div class="algo-desc">Extracts numbers from filenames using regex patterns. '
                                'Primary: page_(\\d+). Fallback: first number in name. '
                                'Example: manuscript_page_003.png → Page 3.</div></div>',
                                unsafe_allow_html=True)
                with sa2:
                    st.markdown('<div class="algo-card"><div class="algo-title">Gap Detection</div>'
                                '<div class="algo-desc">Builds expected set range(min,max+1). '
                                'Missing = expected − actual. Also finds duplicate page numbers '
                                'in the sequence. Flags incomplete batches.</div></div>',
                                unsafe_allow_html=True)
                with sa3:
                    st.markdown('<div class="algo-card"><div class="algo-title">Duplicate Images · dHash</div>'
                                '<div class="algo-desc">Computes a 64-bit perceptual hash per image by '
                                'comparing adjacent pixel brightness. Hamming distance ≤ 10 → duplicate. '
                                'Works even if images are slightly resized or recompressed.</div></div>',
                                unsafe_allow_html=True)

            # ── Page number table ──────────────────────────────
            st.markdown("**📄 Page Number Detection**")
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
            st.dataframe(pd.DataFrame(seq_rows), use_container_width=True, hide_index=True)

            # ── Gap analysis summary ───────────────────────────
            gap = R["sequence"]["gap_analysis"]
            sg1, sg2, sg3, sg4 = st.columns(4)
            sg1.metric("📄 Pages Found", gap["total_detected"])
            sg2.metric("📚 Expected", gap.get("expected_count","—"))
            sg3.metric("❓ Missing", len(gap["missing_pages"]))
            sg4.metric("✅ Complete?", "YES" if gap.get("is_complete") else "NO")

            if gap["missing_pages"]:
                st.warning(f"⚠️ Missing pages: {gap['missing_pages']}")
            if gap["duplicates"]:
                st.warning(f"⚠️ Duplicate page numbers: {gap['duplicates']}")

            # ── Duplicate content section ──────────────────────
            st.markdown("**🔁 Duplicate Image Detection (dHash)**")
            dup_an = R["sequence"]["duplicate_analysis"]
            if dup_an:
                total_dups = dup_an["total_duplicates"]
                if total_dups == 0:
                    st.success("✅ No duplicate images detected across all pages.")
                else:
                    st.warning(f"⚠️ {total_dups} duplicate pair(s) found!")
                    for dp in dup_an["duplicates"]:
                        fn1, fn2 = dp["pair"]
                        st.markdown(
                            f"- **{fn1}** ↔ **{fn2}** "
                            f"(Hamming dist: `{dp['hamming_distance']}`, "
                            f"Similarity: `{dp['similarity_percent']}%`)"
                        )

                # Show hash values
                with st.expander("🔑 Image Hashes (dHash values)"):
                    hash_df = pd.DataFrame(
                        [{"Filename": fn, "dHash": hv}
                         for fn, hv in dup_an["hashes"].items()]
                    )
                    st.dataframe(hash_df, use_container_width=True, hide_index=True)

            st.session_state.pipe_status["sequence_detection"] = "completed"
            pill("✅ Sequence Detection Done", "done")
            pbar.progress(55, text="Sequence detection done")
            time.sleep(0.2)

        except Exception as e:
            st.session_state.pipe_status["sequence_detection"] = "error"
            st.error(f"❌ Sequence detection failed: {e}")
            import traceback; st.code(traceback.format_exc())

    # ══════════════════════════════════════════════════════════
    # STEP 4 — OCR
    # ══════════════════════════════════════════════════════════
    pill("📝 Step 4 · OCR Processing", "run")
    st.session_state.pipe_status["ocr_processing"] = "running"
    pbar.progress(60, text="Running OCR…")

    with st.spinner("Extracting text from images…"):
        try:
            ocrconf = config["ocr"]
            for prep in R["preprocessing"]:
                img = prep.get("processed", prep.get("original"))
                R["ocr"].append(run_ocr(img, ocrconf))

            with st.expander("ℹ️ OCR Algorithm"):
                st.markdown('<div class="algo-card"><div class="algo-title">Tesseract OCR + Fallback</div>'
                            '<div class="algo-desc">Uses Tesseract (--oem 3 --psm 6) for text extraction. '
                            'Reports per-word confidence. Falls back gracefully if Tesseract is not installed. '
                            'Script detection uses Unicode range analysis. Keywords use TF-style frequency counting.</div></div>',
                            unsafe_allow_html=True)

            for i, (prep, ocr_r) in enumerate(zip(R["preprocessing"], R["ocr"])):
                with st.expander(f"📄 {prep['filename']} — confidence: {ocr_r['confidence']}%",
                                 expanded=(i == 0)):
                    oc1, oc2 = st.columns([2, 1])
                    with oc1:
                        st.markdown("**Extracted Text:**")
                        txt = ocr_r["text"][:600] + ("…" if len(ocr_r["text"]) > 600 else "")
                        st.text_area("", txt, height=110, key=f"ocr_{i}", disabled=True)
                    with oc2:
                        st.metric("🎯 Confidence", f'{ocr_r["confidence"]}%')
                        st.metric("📊 Words", ocr_r["word_count"])
                        st.metric("🔤 Script", ocr_r["script"]["script"])
                        if ocr_r["keywords"]:
                            st.markdown("**Top Keywords:**")
                            for w, c in ocr_r["keywords"][:5]:
                                st.markdown(f"- `{w}` ({c}×)")

            st.session_state.pipe_status["ocr_processing"] = "completed"
            pill("✅ OCR Done", "done")
            pbar.progress(75, text="OCR done")
            time.sleep(0.2)

        except Exception as e:
            st.session_state.pipe_status["ocr_processing"] = "error"
            st.error(f"❌ OCR failed: {e}")

    # ══════════════════════════════════════════════════════════
    # STEP 5 — AGGREGATION
    # ══════════════════════════════════════════════════════════
    pill("📊 Step 5 · Aggregation", "run")
    st.session_state.pipe_status["aggregation"] = "running"
    pbar.progress(82, text="Aggregating…")

    with st.spinner("Combining all results…"):
        try:
            R["aggregated"] = aggregate_results(
                R["preprocessing"], R["quality"], R["sequence"], R["ocr"],
                config["aggregation"])
            R["batch_summary"] = generate_batch_summary(R["aggregated"])

            bs = R["batch_summary"]
            ba1,ba2,ba3,ba4,ba5 = st.columns(5)
            ba1.metric("📷 Total", bs["total_images"])
            ba2.metric("✅ Ready", bs["ready"])
            ba3.metric("⚠️ Review", bs["review"])
            ba4.metric("❌ Rejected", bs["rejected"])
            ba5.metric("📈 Pass Rate", f'{bs["pass_rate"]}%')

            # Final table with coloured status
            st.markdown("**📋 Final Results Table**")
            final_df = pd.DataFrame(R["aggregated"])
            final_df.columns = ["Filename","Quality","Page#","OCR Conf.","Final Score","Status","Blur","Skew"]
            st.dataframe(final_df, use_container_width=True, hide_index=True)

            st.session_state.pipe_status["aggregation"] = "completed"
            pill("✅ Aggregation Done", "done")
            pbar.progress(92, text="Aggregating done")
            time.sleep(0.2)

        except Exception as e:
            st.session_state.pipe_status["aggregation"] = "error"
            st.error(f"❌ Aggregation failed: {e}")

    # ══════════════════════════════════════════════════════════
    # STEP 6 — OUTPUT
    # ══════════════════════════════════════════════════════════
    pill("💾 Step 6 · Output Generation", "run")
    st.session_state.pipe_status["output_generation"] = "running"
    pbar.progress(96, text="Generating reports…")

    with st.spinner("Saving reports…"):
        try:
            out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   config["output"]["output_dir"])
            os.makedirs(out_dir, exist_ok=True)
            csv_path  = os.path.join(out_dir, config["output"]["csv_filename"])
            json_path = os.path.join(out_dir, config["output"]["json_filename"])
            R["csv_path"]  = generate_csv(R["aggregated"], csv_path)
            R["json_path"] = generate_json_audit(R["aggregated"], R["batch_summary"], json_path)

            dc1, dc2 = st.columns(2)
            if R["csv_path"] and os.path.exists(R["csv_path"]):
                with open(R["csv_path"], "r", encoding="utf-8") as f:
                    csv_data = f.read()
                dc1.download_button("📥 Download CSV Report",
                                    data=csv_data,
                                    file_name="processing_results.csv",
                                    mime="text/csv", use_container_width=True)
            if R["json_path"] and os.path.exists(R["json_path"]):
                with open(R["json_path"], "r", encoding="utf-8") as f:
                    json_data = f.read()
                dc2.download_button("📥 Download JSON Audit Log",
                                    data=json_data,
                                    file_name="audit_log.json",
                                    mime="application/json", use_container_width=True)

            st.session_state.pipe_status["output_generation"] = "completed"
            pill("✅ Output Done", "done")
            pbar.progress(100, text="Pipeline complete ✅")

        except Exception as e:
            st.session_state.pipe_status["output_generation"] = "error"
            st.error(f"❌ Output failed: {e}")

    st.session_state.results = R
    st.session_state.running = False
    st.balloons()
    st.markdown("""
    <div style="text-align:center;padding:1.5rem;background:linear-gradient(135deg,#1a4d2e,#0e1117);
    border-radius:14px;border:1px solid #2a4d3e;margin-top:1rem;">
      <h2 style="color:#00c853;margin:0;">🎉 Pipeline Completed Successfully!</h2>
      <p style="color:#8892a6;margin-top:0.4rem;">All 6 modules finished. Download your reports above.</p>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# IDLE STATE
# ═══════════════════════════════════════════════════════════════
if not run_btn and st.session_state.results is None:
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
### 🚀 Quick Start
1. Click **Generate Sample Images** in the sidebar
2. Click **Run Full Pipeline**
3. Watch all 6 modules execute live
4. Browse images, inspect skew charts, check duplicates
5. Download CSV + JSON reports
        """)
    with c2:
        st.markdown("""
### 📋 Modules & Algorithms
| # | Module | Algorithm |
|---|--------|-----------|
| 1 | Preprocessing | Grayscale · Resize · NLM · CLAHE |
| 2 | Quality Check | Laplacian · **PPM Skew** · Margin · Dark% |
| 3 | Sequence + Dups | Regex · **dHash** · Hamming distance |
| 4 | OCR | Tesseract · Confidence · Script detect |
| 5 | Aggregation | Weighted scoring · READY/REVIEW/REJECT |
| 6 | Output | CSV + JSON audit log |
        """)
    st.info("👆 Click **Generate Sample Images** then **Run Full Pipeline** to start.")
