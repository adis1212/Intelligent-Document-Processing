# 📊 Module Status Report — IDP Manuscript Processing System

> **Last Updated:** April 2026  
> **Project:** Intelligent Document Processing for Heritage Manuscript Archives

---

## ✅ WORKING MODULES

### ✅ Module 1 — Preprocessor (`modules/preprocessor.py`)
**Status: FULLY WORKING**

| Function | Status | Details |
|----------|--------|---------|
| `load_batch()` | ✅ Working | Loads all images from folder, supports JPG/PNG/TIFF/BMP |
| `preprocess_image()` | ✅ Working | 4-step pipeline: grayscale → resize → denoise → CLAHE |

**How it works:**
1. **Grayscale** — `cv2.cvtColor(img, COLOR_BGR2GRAY)` converts 3-channel RGB to single-channel. Reduces data volume by 3× while preserving luminance/texture information essential for quality analysis.
2. **Resize** — `cv2.resize(..., INTER_AREA)` scales the image to target resolution using area interpolation (best quality for downscaling). Ensures all images are consistent size.
3. **Denoise (NLM)** — `cv2.fastNlMeansDenoising()` uses Non-Local Means: for each pixel, finds similar patches in a search window and averages them. Removes scan noise while preserving document edges — superior to Gaussian blur.
4. **CLAHE** — Contrast Limited Adaptive Histogram Equalisation divides image into tiles (8×8), equalises each tile's histogram independently, and clips at `clipLimit=2.0` to prevent over-amplification of noise.

---

### ✅ Module 2 — Quality Checker (`modules/quality_checker.py`)
**Status: FULLY WORKING with IMPROVED Skew Detection**

| Check | Algorithm | Status |
|-------|-----------|--------|
| Blur Detection | Laplacian Variance | ✅ Working |
| Skew Detection | **Projection Profile Method (PPM)** | ✅ Working |
| Crop Detection | Border Margin Analysis | ✅ Working |
| Occlusion Detection | Dark Pixel Ratio | ✅ Working |
| Quality Score | Weighted Composite | ✅ Working |

**How each algorithm works:**

#### 🔍 Blur Detection — Laplacian Variance
```
Input Image → Apply Laplacian Operator → Compute Variance → Compare to threshold
```
- The Laplacian (`∇²f = ∂²f/∂x² + ∂²f/∂y²`) measures the second derivative of pixel intensity.
- Sharp images have many strong edges → high Laplacian response → **high variance**.
- Blurry images have smoothed edges → low Laplacian response → **low variance**.
- **Threshold:** Below 100 = BLURRED.

#### 📐 Skew Detection — Projection Profile Method (PPM)
```
Binary Image → Rotate by angle θ → Row Sums → Variance → Find θ with max variance
```
- **Step 1:** Convert to binary (Otsu threshold) — dark text on white background.
- **Step 2:** Test 81 angles from −20° to +20°.
- **Step 3:** For each angle, rotate the binary image and compute row projection (sum of dark pixels per row).
- **Step 4:** When text rows are horizontal, the projection profile has clear peaks (text rows) and valleys (whitespace) → **maximum variance**.
- **Step 5:** The angle producing maximum variance is the document's rotation angle.
- **Output:** Detected angle + deskewed corrected image + projection chart.
- **Threshold:** Magnitude > 3° = SKEWED.

#### 📦 Crop Detection — Border Margin Analysis
```
Binary Image → Check 2% border strips → Content at edges → CROPPED
```
- Binarizes image with Otsu threshold.
- Checks 4 border strips (top/bottom/left/right = 2% of dimension).
- If all 4 borders have dark content → image is cropped to the edge.

#### 🌑 Occlusion Detection — Dark Pixel Ratio
```
Grayscale Image → Count pixels < 30 → dark% > 15% → OCCLUDED
```
- Very dark pixels (intensity < 30) indicate shadows, stamps, thumbprints, or water damage.
- If more than 15% of the image area is excessively dark → flagged as occluded.

---

### ✅ Module 3 — Sequence Detector (`modules/sequence_detector.py`)
**Status: FULLY WORKING with Duplicate Detection**

| Function | Algorithm | Status |
|----------|-----------|--------|
| `extract_page_from_filename()` | Regex matching | ✅ Working |
| `detect_gaps()` | Set difference | ✅ Working |
| `compute_dhash()` | Difference Hash | ✅ Working |
| `hamming_distance()` | Bit XOR + count | ✅ Working |
| `detect_duplicates()` | dHash + Hamming | ✅ Working |

**How each algorithm works:**

#### 🔢 Page Number Extraction — Regex
```python
re.search(r"page_(\d+)", filename)  # Primary pattern
re.search(r"(\d+)", filename)        # Fallback: any number
```
- Extracts page number from filename without opening the image.
- Fast and reliable for well-named files.

#### ❓ Gap Detection — Set Difference
```python
expected = set(range(min_page, max_page + 1))
missing  = sorted(expected - actual_pages)
```
- Builds the complete expected set of page numbers.
- Subtracts the actual detected pages to find gaps.

#### 🔁 Duplicate Detection — **dHash (Difference Hash)**
```
Image → Resize (9×8) → Grayscale → Compare adjacent pixels → 64-bit hash
```
- **Step 1:** Resize to 9×8 pixels (ignoring detail).
- **Step 2:** For each of the 8 rows, compare 8 adjacent pixel pairs: bit = 1 if left > right.
- **Step 3:** Concatenate 64 bits → single integer hash.
- **Step 4:** Compare any two images: `XOR hashes → count differing bits (Hamming distance)`.
- **Interpretation:**
  - Hamming distance = 0 → Identical images
  - Hamming distance ≤ 10 → Very similar / near-duplicate
  - Hamming distance > 20 → Different images

---

### ✅ Module 4 — OCR Engine (`modules/ocr_engine.py`)
**Status: WORKING (with fallback if Tesseract not installed)**

| Function | Status | Details |
|----------|--------|---------|
| `extract_text()` | ✅ Working | Uses pytesseract; graceful fallback mode |
| `extract_keywords()` | ✅ Working | TF-style frequency count, stop-word filtered |
| `detect_script()` | ✅ Working | Unicode range analysis |

**How it works:**
- **Tesseract OCR:** Uses `--oem 3 --psm 6` (LSTM engine, uniform text block).
- **Confidence:** `image_to_data()` returns per-word confidence; averaged across words.
- **Keywords:** Words are lowercased, stop words removed, then counted by frequency.
- **Script Detection:** Counts Unicode code point ranges (Latin U+0041-U+007A, Devanagari U+0900-U+097F, etc.).

**⚠️ Note:** Requires Tesseract to be installed separately (`choco install tesseract` on Windows). Without it, runs in fallback mode returning placeholder text.

---

### ✅ Module 5 — Aggregator (`modules/aggregator.py`)
**Status: FULLY WORKING**

**How it works:**
- Combines quality_score (40%), OCR confidence (40%), and page detection bonus (20%).
- Assigns READY (≥70), REVIEW (40–69), or REJECT (<40) status.
- Generates batch-level statistics (total, ready, review, rejected, pass rate).

---

### ✅ Module 6 — Output Generator (`modules/output_generator.py`)
**Status: FULLY WORKING**

| Output | Status | Details |
|--------|--------|---------|
| CSV Report | ✅ Working | `outputs/processing_results.csv` |
| JSON Audit | ✅ Working | `outputs/audit_log.json` with timestamp + metadata |
| Download buttons | ✅ Working | In Streamlit UI |

---

## ⚠️ PARTIALLY WORKING

### ⚠️ Module 4 — OCR (Without Tesseract)
**Status: FALLBACK MODE**

- **Problem:** Tesseract binary not installed on the system.
- **Effect:** `run_ocr()` returns empty text with 0% confidence.
- **Workaround:** All other modules work fine. OCR results show `[OCR Fallback]` message.
- **Fix:** Install Tesseract:
  ```bash
  # Windows (Chocolatey)
  choco install tesseract
  
  # Or download from: https://github.com/UB-Mannheim/tesseract/wiki
  # Add to PATH after installation
  ```

---

## ❌ NOT YET IMPLEMENTED

| Feature | Module | Planned Phase |
|---------|--------|---------------|
| OCR-based page number detection | Sequence Detector | Phase 5 extension |
| Auto deskewing (apply to pipeline) | Preprocessor | Phase 4 improvement |
| Multi-language manuscript support | OCR Engine | Phase 6 |
| Historical script detection (Arabic, Devanagari) | OCR Engine | Phase 6 |
| Visual report (HTML export) | Output Generator | Phase 8 |
| REST API endpoint | main.py | Phase 8 |

---

## 🔧 How to Run Everything

### Launch Dashboard
```bash
cd idp_manuscript
streamlit run dashboard.py
# Opens at http://localhost:8501
```

### Generate Test Images
```bash
python -c "from utils.sample_generator import generate_sample_manuscripts; generate_sample_manuscripts('data/sample_images', 6)"
```

### Run Module Tests
```bash
python run_tests.py
# Results written to test_results.txt
```

### Run CLI Pipeline
```bash
python main.py data/sample_images
```

---

## 📦 Dependency Status

| Package | Required | Installed | Notes |
|---------|----------|-----------|-------|
| streamlit | ✅ | ✅ | Dashboard UI |
| opencv-python | ✅ | ✅ | All image processing |
| numpy | ✅ | ✅ | Array operations |
| Pillow | ✅ | ✅ | Image I/O to Streamlit |
| pandas | ✅ | ✅ | Result tables |
| pytesseract | ✅ | ✅ | OCR wrapper (fallback if tesseract binary missing) |
| PyYAML | ✅ | ✅ | Config loading |
| scikit-image | ✅ | ✅ | Additional image utils |
| matplotlib | ✅ | ✅ | Skew projection chart |
| **tesseract binary** | ⚠️ | ❓ | External install needed for full OCR |
