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
| 4 | **OCR Processing** | Extract text, compute confidence, detect script |
| 5 | **Aggregation** | Combine scores → READY / REVIEW / REJECT |
| 6 | **Output Generation** | Export CSV report + JSON audit log |

---

## ✅ Module 1: Preprocessing — COMPLETED

### What it does
- **Loads** all images from a specified folder (JPG, PNG, TIFF, BMP)
- **Converts** to grayscale for uniform processing
- **Resizes** images to a target resolution (configurable)
- **Denoises** using Non-Local Means algorithm (removes scan artifacts)
- **Enhances contrast** using CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Displays** before vs. after preview in the UI

### Key functions
```python
load_batch(folder_path)        # Load all images from a directory
preprocess_image(image, config) # 4-step preprocessing pipeline
```

### Test Results ✅
```
Loaded: 6 sample manuscript images
Preprocessing pipeline: grayscale → resize → denoise → enhance
Success: True | Original: (800, 1000) → Processed: (800, 1000)
```

---

## ✅ Module 2: Quality Checker — COMPLETED

### What it does
- **Blur Detection** — Laplacian variance (below threshold = BLURRED)
- **Skew Detection** — Hough Line Transform (angle > 5° = SKEWED)
- **Crop Detection** — Margin percentage check (no margins = CROPPED)
- **Occlusion Detection** — Dark pixel area analysis
- **Quality Score** — Weighted composite score (0–100)

### Score Breakdown
| Check | Weight | Method |
|-------|--------|--------|
| Blur | 40 pts | Laplacian variance |
| Skew | 20 pts | Hough lines angle |
| Crop | 20 pts | Border margin check |
| Occlusion | 20 pts | Dark pixel ratio |

---

## ✅ Module 3: Sequence Detector — COMPLETED

- Extracts page numbers from filenames using regex
- Detects **missing pages** in the sequence
- Detects **duplicate pages**
- Supports fallback patterns for non-standard filenames

---

## ✅ Module 4: OCR Engine — COMPLETED

- Text extraction via **pytesseract** (with graceful fallback if not installed)
- Per-word **confidence scoring**
- **Keyword extraction** (top-N words after stop-word filtering)
- **Script detection** (Latin, Devanagari, Arabic, CJK)

---

## ✅ Module 5: Aggregator — COMPLETED

Combines all module scores into a final decision:

| Final Score | Status |
|-------------|--------|
| ≥ 70 | ✅ READY |
| 40–69 | ⚠️ REVIEW |
| < 40 | ❌ REJECT |

---

## ✅ Module 6: Output Generator — COMPLETED

- **CSV Report** — `outputs/processing_results.csv`
- **JSON Audit Log** — `outputs/audit_log.json` (with timestamp + metadata)
- **Download buttons** in the Streamlit UI

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
1. Click **"Generate Sample Images"** (sidebar) — creates 6 test manuscripts
2. Click **"Run Pipeline"** — watch all 6 modules execute live
3. View results tables, image previews, quality scores
4. Download **CSV** and **JSON** reports

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
│   ├── quality_checker.py    ← Module 2: Quality analysis
│   ├── sequence_detector.py  ← Module 3: Page sequencing
│   ├── ocr_engine.py         ← Module 4: Text extraction
│   ├── aggregator.py         ← Module 5: Score aggregation
│   └── output_generator.py   ← Module 6: CSV + JSON output
│
├── utils/
│   ├── config_loader.py      ← YAML config with caching
│   ├── logger.py             ← Structured logging
│   └── sample_generator.py  ← Synthetic manuscript image creator
│
├── data/
│   └── sample_images/        ← Input images (auto-generated for demo)
│
├── outputs/                  ← Generated reports (CSV + JSON)
│
└── tests/
    └── test_modules.py       ← Unit tests for all modules
```

---

## ⚙️ Configuration (`config.yaml`)

All thresholds are tunable without changing code:

```yaml
quality:
  blur_threshold: 100.0      # Laplacian variance below = BLURRED
  skew_threshold: 5.0        # Degrees above = SKEWED
  min_quality_score: 50.0    # Minimum pass score

ocr:
  language: "eng"            # Tesseract language
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
| OpenCV | Image processing (blur, skew, CLAHE) |
| NumPy | Array operations |
| Pillow | Image I/O |
| pytesseract | OCR text extraction |
| pandas | Tabular result display |
| PyYAML | Configuration management |
| scikit-image | Additional image utilities |

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
manuscript_page_001.png,78.5,1,65.2,74.3,READY
manuscript_page_002.png,45.0,2,40.1,42.0,REVIEW
manuscript_page_003.png,22.1,3,18.5,20.8,REJECT
```

### JSON Audit Log
```json
{
  "metadata": {
    "generated_at": "2026-04-01T00:00:00",
    "system": "IDP Manuscript Processing System",
    "version": "1.0.0"
  },
  "batch_summary": {
    "total_images": 6,
    "ready": 4,
    "review": 1,
    "rejected": 1,
    "pass_rate": 66.7
  }
}
```

---

## 🎓 Academic Context

**Project:** Intelligent Document Processing for Heritage Manuscript Archives  
**Domain:** Document AI / Computer Vision  
**Techniques:** Image preprocessing, quality metrics, OCR, sequence analysis  

### Key Algorithms
- **Laplacian Variance** — Measures image sharpness (blur detection)
- **Hough Line Transform** — Detects dominant line angles (skew detection)
- **CLAHE** — Adaptive contrast enhancement preserving local details
- **Non-Local Means Denoising** — Edge-preserving noise removal
- **Laplacian of Gaussian** — Edge detection for crop analysis

---

*Built with Python | OpenCV | Streamlit*