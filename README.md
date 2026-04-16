# 📜 Intelligent Document Processing (IDP) for Heritage Manuscript Archives

> A production-grade, modular AI pipeline for digitizing, analyzing, and archiving heritage manuscript images — built with Python, OpenCV, and Streamlit.

---

## 🎯 Project Overview

This system automates the processing of scanned manuscript images through a **6-stage intelligent pipeline**:

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | **Preprocessing** | Load images, resize, denoise, contrast-enhance |
| 2 | **Quality Check** | Detect blur, skew, cropping issues, occlusion |
| 3 | **Sequence Detection** | Extract page numbers, detect gaps & duplicates |
| 4 | **OCR Processing** | Extract English text, compute confidence, detect script |
| 5 | **Aggregation & Search** | Combine scores, weighted scoring, document search |
| 6 | **Output Generation** | Export CSV report + JSON audit log |

---

## ✨ Key Features

- **Upload Your Own Files** — Browse and select manuscript images directly from your computer
- **6-Step Pipeline** — Preprocessing → Quality Check → Sequence Detection → OCR → Aggregation → Output
- **Skew Detection with Override** — Skewed pages are automatically capped at REVIEW status
- **Smart OCR Engine** — Extracts English text with quality-derived confidence scores (65–92%)
- **Document Search** — Full-text search across extracted text, filenames, and keywords
- **Color-Coded Dashboard** — White background, black text, colorful metric numbers for instant readability
- **Download Reports** — Export results as CSV and JSON audit log

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd idp_manuscript
pip install -r requirements.txt
```

### 2. Launch the Dashboard
```bash
streamlit run dashboard.py
```

### 3. In the Browser (http://localhost:8501)
1. **Upload** your manuscript images using the file picker in the sidebar
2. Click **"Run Full Pipeline"** — watch all 6 modules execute with progress bar
3. Browse the image gallery and inspect individual pages
4. View quality scores, OCR text, skew detection results
5. Search across all extracted text and keywords
6. Download **CSV** and **JSON** reports

> 💡 For quick testing, expand **"Or use sample images"** in the sidebar to generate 6 synthetic manuscript pages.

---

## 📁 Project Structure

```
idp_manuscript/
├── dashboard.py              ← Streamlit UI (main entry point)
├── main.py                   ← CLI pipeline runner
├── config.yaml               ← All thresholds & settings
├── requirements.txt          ← Python dependencies
│
├── modules/
│   ├── preprocessor.py       ← Module 1: Image preprocessing
│   ├── quality_checker.py    ← Module 2: Quality analysis (blur, skew, crop, occlusion)
│   ├── sequence_detector.py  ← Module 3: Page sequencing & duplicate detection
│   ├── ocr_engine.py         ← Module 4: Text extraction & keyword analysis
│   ├── aggregator.py         ← Module 5: Score aggregation & search index
│   └── output_generator.py   ← Module 6: CSV + JSON output
│
├── utils/
│   ├── config_loader.py      ← YAML config with caching
│   ├── logger.py             ← Structured logging
│   └── sample_generator.py   ← Synthetic manuscript image creator
│
├── data/
│   ├── uploads/              ← User-uploaded images (processed here)
│   └── sample_images/        ← Auto-generated sample images
│
├── outputs/                  ← Generated reports (CSV + JSON)
│
└── tests/
    └── test_modules.py       ← Unit tests for all modules
```

---

## ✅ Module Details

### Module 1: Preprocessing
- **Loads** all images from folder (JPG, PNG, TIFF, BMP)
- **Converts** to grayscale → **Resizes** → **Denoises** (NLM) → **Enhances** (CLAHE)

### Module 2: Quality Checker
| Check | Weight | Algorithm |
|-------|--------|-----------|
| Blur | 40 pts | Laplacian variance |
| Skew | 20 pts | Projection Profile Method (PPM) |
| Crop | 20 pts | Border margin analysis |
| Occlusion | 20 pts | Dark pixel ratio |

**Skew Override**: If skew exceeds threshold (default 5°), page status is automatically capped at REVIEW.

### Module 3: Sequence Detection
- Extracts page numbers from filenames via regex
- Detects **missing pages** and **duplicate page numbers**
- **dHash** perceptual hashing for duplicate image detection (Hamming distance ≤ 10)

### Module 4: OCR Engine
- Extracts English text from manuscript images
- Confidence scores derived from image quality metrics (Laplacian + intensity)
- **Keyword extraction** via word frequency analysis
- **Script detection** (Latin, Devanagari, Arabic, CJK) via Unicode range classification

### Module 5: Aggregation & Search
- **Weighted Scoring**: Quality (40%) + OCR Confidence (40%) + Sequence (20%)
- **Status Thresholds**: ≥70 = READY ✅ | 40–69 = REVIEW ⚠️ | <40 = REJECT ❌
- **Document Search**: Inverted index over OCR text, filenames, keywords, page numbers

### Module 6: Output Generation
- **CSV Report** — `outputs/processing_results.csv`
- **JSON Audit Log** — `outputs/audit_log.json` (with timestamps & batch metadata)

---

## ⚙️ Configuration (`config.yaml`)

All thresholds are tunable without changing code:

```yaml
quality:
  blur_threshold: 100.0      # Laplacian variance below = BLURRED
  skew_threshold: 5.0        # Degrees above = SKEWED
  min_quality_score: 50.0    # Minimum pass score

ocr:
  language: "eng"            # OCR language
  confidence_threshold: 60.0 # Minimum OCR confidence

aggregation:
  ready_threshold: 70.0      # Score >= this = READY
  review_threshold: 40.0     # Score >= this = REVIEW
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.8+ | Core language |
| Streamlit | Interactive web dashboard |
| OpenCV | Image processing (blur, skew, CLAHE, denoising) |
| NumPy | Array operations |
| Pillow | Image I/O |
| pandas | Tabular result display |
| PyYAML | Configuration management |
| scikit-image | Additional image utilities |
| matplotlib | Projection profile charts |

---

## 🧪 Running Tests

```bash
cd idp_manuscript
python run_tests.py
```

Expected output:
```
TEST 1: Preprocessor ......... PASS
TEST 2: Quality Checker ....... PASS
TEST 3: Sequence Detector ..... PASS
TEST 4: OCR Engine ............ PASS
TEST 5: Aggregator ............ PASS
TEST 6: Output Generator ...... PASS
ALL MODULE TESTS PASSED!
```

---

## 📊 Sample Output

### CSV Report
```
filename,quality_score,page_number,ocr_confidence,final_score,status
manuscript_page_001.png,95,1,82.7,91.1,READY
manuscript_page_003.png,40.3,3,67.7,63.2,REVIEW
manuscript_page_005.png,71.2,5,85.0,82.5,REVIEW
```

### Dashboard Summary Cards
| Metric | Example Value | Color |
|--------|--------------|-------|
| Total Pages | 6 | 🔵 Blue |
| Ready | 4 | 🟢 Green |
| Need Review | 2 | 🟠 Orange |
| Rejected | 0 | 🔴 Red |
| Pass Rate | 66.7% | 🟣 Purple |

---

## 🎓 Academic Context

**Project:** Intelligent Document Processing for Heritage Manuscript Archives  
**Domain:** Document AI / Computer Vision  
**Techniques:** Image preprocessing, quality metrics, OCR, sequence analysis, document search

### Key Algorithms
- **Laplacian Variance** — Measures image sharpness (blur detection)
- **Projection Profile Method (PPM)** — Row-sum variance across rotation angles (skew detection)
- **dHash (Difference Hash)** — 64-bit perceptual hash for duplicate image detection
- **CLAHE** — Adaptive contrast enhancement preserving local details
- **Non-Local Means Denoising** — Edge-preserving noise removal
- **Inverted Index** — Full-text search across OCR output and metadata

---

*Built with Python | OpenCV | Streamlit*