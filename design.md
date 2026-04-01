# IDP for Heritage Manuscript Archives
## Technical Design Document

**Project:** Intelligent Document Processing (IDP) for Heritage Manuscript Archives
**Team:** CSE – Data Science, VIT
**Version:** 1.0

---

## 1. System Architecture Summary

The system follows a linear pipeline architecture with 7 swimlanes matching the eraser.io architecture diagram. All three processing modules (Quality, Sequence, OCR) run on each image independently. Their outputs converge at the Aggregation layer which makes the final routing decision.

```
INPUT → PREPROCESSING → [QUALITY | SEQUENCE | OCR] → AGGREGATION → OUTPUT
                                                              ↓
                                                      HUMAN REVIEW (low confidence)
```

---

## 2. Technology Stack

| Layer | Component | Library / Tool | Version |
|-------|-----------|----------------|---------|
| Image I/O | Load, save, resize | `Pillow` | ≥ 9.0 |
| Computer Vision | Blur, skew, occlusion | `opencv-python` | ≥ 4.7 |
| OCR Engine | Text extraction | `pytesseract` + Tesseract | ≥ 5.0 |
| Perceptual Hash | Duplicate detection | `imagehash` | ≥ 4.3 |
| NLP | Text cleanup, tokenize | `spaCy` | ≥ 3.5 |
| Keyword Extraction | RAKE algorithm | `rake-nltk` | ≥ 1.0 |
| TF-IDF Re-ranking | Keyword scoring | `scikit-learn` | ≥ 1.2 |
| Data Handling | CSV, DataFrames | `pandas` | ≥ 1.5 |
| Dashboard | Web UI | `streamlit` | ≥ 1.20 |
| Config | YAML parsing | `PyYAML` | ≥ 6.0 |
| Progress | Batch progress bar | `tqdm` | ≥ 4.0 |
| Numerics | Array math | `numpy` | ≥ 1.23 |

---

## 3. Config File Design

**File:** `config.yaml`

```yaml
pipeline:
  input_formats: [".jpg", ".jpeg", ".png", ".tiff", ".tif"]
  resize_max_px: 2000
  output_dir: "./outputs"

quality:
  blur_threshold: 100          # Laplacian variance below this = BLURRED
  skew_max_angle: 3.0          # Degrees beyond this = SKEWED
  occlusion_area_pct: 5.0      # % of image area for dark blob = OCCLUDED
  crop_border_px: 10           # Border strip to check for content cutoff

sequence:
  phash_threshold: 8           # Hamming distance <= this = DUPLICATE
  ocr_confidence_min: 0.7      # Minimum to accept a layer's page number
  year_range: [1600, 2025]     # Filter range for year detection
  tesseract_psm_region: 7      # PSM mode for region OCR (single text line)
  tesseract_psm_full: 3        # PSM mode for full page OCR

ocr:
  languages: "hin+san+eng"
  top_keywords: 10
  min_word_confidence: 60      # Tesseract word conf threshold (0-100)

aggregation:
  weight_quality: 0.40
  weight_sequence: 0.35
  weight_ocr: 0.25
  review_threshold: 0.50       # Below this score → human review
```

---

## 4. Data Schema

### 4.1 Per-image record (internal dict, passed between modules)

```python
{
  # Identifiers
  "filename": "39(1907-1915)_001.jpg",
  "filepath": "/data/batch1/39(1907-1915)_001.jpg",
  "batch_id": "batch1",

  # Module 1 — Quality
  "quality_status": "OK",           # OK/BLURRED/SKEWED/CROPPED/OCCLUDED/MULTI_FLAG/ERROR
  "quality_score": 0.87,            # 0.0 – 1.0
  "blur_variance": 142.3,
  "skew_angle": 1.2,
  "is_cropped": False,
  "cropped_edges": [],
  "is_occluded": False,
  "occlusion_area_pct": 0.0,

  # Module 2 — Sequence
  "detected_page_no": 1,            # int or None
  "detection_method": "filename",   # filename/region_ocr/full_ocr/interpolated/unknown
  "sequence_confidence": 0.90,
  "sequence_status": "OK",          # OK/MISSING/DUPLICATE/EXPECTED_BLANK/INTERPOLATED/UNKNOWN
  "duplicate_of": None,             # filename string or None
  "hamming_distance": None,

  # Module 3 — OCR + NLP
  "ocr_confidence": 0.74,
  "detected_script": "Devanagari",  # Devanagari/Latin/Mixed/Unknown
  "extracted_text": "...",
  "top_keywords": ["keyword1", "keyword2"],

  # Aggregation
  "final_score": 0.83,
  "pipeline_status": "results_ready",   # results_ready/human_review
  "routing_reason": None,
  "notes": ""
}
```

### 4.2 Output CSV columns

```
filename | detected_page_no | detection_method | quality_status | quality_score |
sequence_status | sequence_confidence | ocr_confidence | detected_script |
top_keywords | final_score | pipeline_status | notes
```

### 4.3 Audit log JSON structure

```json
{
  "run_id": "2026-03-26T14:23:00",
  "batch": "batch1",
  "total_images": 120,
  "total_flagged": 18,
  "human_review_queue": [
    {
      "filename": "page_042.jpg",
      "final_score": 0.38,
      "routing_reason": "blur_score=0.21 ocr_confidence=0.41",
      "quality_details": { ... },
      "sequence_details": { ... },
      "ocr_details": { ... },
      "timestamp": "2026-03-26T14:25:11",
      "resolved": false,
      "resolution_notes": ""
    }
  ]
}
```

---

## 5. Module Design

### 5.1 Module 1 — Image Quality Checker

**File:** `modules/quality_checker.py`

#### Blur Detection Algorithm
```
INPUT: grayscale numpy array
1. Apply cv2.Laplacian(image, cv2.CV_64F)
2. Compute variance of Laplacian output
3. score = min(variance / blur_threshold, 1.0)
4. is_blurred = (variance < blur_threshold)
OUTPUT: is_blurred (bool), blur_score (float), variance (float)
```

**Why Laplacian variance:** The Laplacian is a second-order derivative operator that highlights regions of rapid intensity change (edges). A sharp image has many strong edges → high variance. A blurry image has soft transitions → low variance. This is the standard and most reliable single-metric blur estimator.

**Edge case — very dark images:** A very dark image can have low variance due to darkness, not blur. Guard: if mean pixel intensity < 30, flag as DARK_IMAGE instead of BLURRED.

#### Skew Detection Algorithm
```
INPUT: grayscale numpy array
1. Apply Canny edge detection: cv2.Canny(image, 50, 150)
2. Run Hough Line Transform: cv2.HoughLinesP(edges, ...)
3. For each detected line, compute angle from horizontal
4. Take median angle across all detected lines
5. is_skewed = (abs(median_angle) > skew_max_angle)
OUTPUT: is_skewed (bool), skew_angle (float in degrees)
```

**Edge case — handwritten manuscripts with many diagonal strokes:** Devanagari has many near-vertical strokes that can produce high-angle detections. Filter: discard lines shorter than image_width × 0.1 before angle computation.

#### Crop Detection Algorithm
```
INPUT: grayscale numpy array, border_px from config
1. Extract top strip: image[0:border_px, :]
2. Extract bottom strip: image[-border_px:, :]
3. Extract left strip: image[:, 0:border_px]
4. Extract right strip: image[:, -border_px:]
5. For each strip: if mean_intensity < 15 AND std < 5 → solid black border → likely cropped
   OR if mean_intensity > 240 AND std < 5 → solid white strip → content cut off
6. Return which edges are affected
OUTPUT: is_cropped (bool), affected_edges (list of 'top'/'bottom'/'left'/'right')
```

#### Occlusion Detection Algorithm
```
INPUT: grayscale numpy array
1. Compute median pixel intensity of entire image (background estimate)
2. Threshold: pixels < (median - 80) → potential dark occluder
3. Apply morphological opening to remove thin text strokes
4. Find contours in thresholded image
5. For each contour:
   - compute area as % of total image area
   - check shape irregularity (not a straight line = not manuscript line)
   - if area > occlusion_area_pct AND irregular shape → flag
OUTPUT: is_occluded (bool), occlusion_area_pct (float)
```

**Edge case — manuscript borders and decorative frames:** Some manuscripts have dark ornamental borders. Guard: ignore contours touching image edges (likely border, not occlusion).

---

### 5.2 Module 2 — Sequence and Duplicate Detection

**File:** `modules/sequence_detector.py`

#### Layer 1 — Filename Parser
```
INPUT: filename string  e.g. "39(1907-1915)_folio_005.jpg"
1. Remove extension
2. Find all numeric tokens: re.findall(r'\d+', filename)
3. For each token:
   a. If 4-digit AND 1600 ≤ value ≤ 2025 → discard (year)
   b. If preceded by non-digit that is immediately followed by '(' → discard (manuscript ID)
   c. Remaining tokens: take last surviving numeric token as candidate page number
4. confidence = 0.90 if candidate found, else 0.0
OUTPUT: page_number (int or None), confidence (float)
```

**Filename examples:**
- `39(1907-1915)_005.jpg` → tokens: [39, 1907, 1915, 005] → filter 1907, 1915 (years), filter 39 (before bracket) → page_no = 5, conf = 0.90
- `MS_042_recto.jpg` → tokens: [042] → page_no = 42, conf = 0.90
- `scan_batch.jpg` → tokens: [] → page_no = None, conf = 0.0

#### Devanagari Numeral Normalizer
```python
DEVA_MAP = {
    '०':'0','१':'1','२':'2','३':'3','४':'4',
    '५':'5','६':'6','७':'7','८':'8','९':'9'
}
def normalize_numerals(text):
    for deva, arabic in DEVA_MAP.items():
        text = text.replace(deva, arabic)
    return text
```

Apply this to ALL OCR output before any numeric extraction.

#### Layer 2 — Region OCR
```
INPUT: image_path, config
1. Load preprocessed image
2. Crop top 20%: image[0 : int(h*0.20), :]
3. Crop bottom 20%: image[int(h*0.80) : h, :]
4. For each crop:
   a. Run pytesseract.image_to_string(crop, config='--psm 7 -l hin+san+eng')
   b. Normalize Devanagari numerals
   c. Extract all isolated numeric tokens (not part of words)
   d. Filter: keep only tokens with 1–4 digits
5. Return best candidate with confidence estimate from Tesseract
OUTPUT: page_number (int or None), confidence (float)
```

#### Layer 3 — Full-Page OCR + Outlier Filter
```
INPUT: image_path, batch_y_cluster (pre-computed y-band dict)
1. Run pytesseract.image_to_data() → returns word-level DataFrame
2. Normalize Devanagari in all text values
3. Filter rows: text matches r'^\d{1,4}$' (isolated 1–4 digit tokens)
4. Score each candidate:
   - y_band_score: 1.0 if y-position within ±30px of batch_y_cluster, else 0.3
   - isolation_score: 1.0 if no adjacent words within 50px, else 0.5
   - conf_score: tesseract_word_conf / 100
   - total_score = mean(y_band_score, isolation_score, conf_score)
5. Return highest scoring candidate
OUTPUT: page_number (int or None), confidence (float)

Pre-compute batch_y_cluster:
- Run full OCR on all images in batch (or a sample of 20%)
- Collect y-coordinates of all isolated numeric tokens
- Use histogram peak as the dominant y-band for page numbers
```

#### pHash Duplicate Detector
```
INPUT: list of image_paths, hamming_threshold
1. For each image: compute pHash using imagehash.phash(Image.open(path))
2. Store in dict: {filename: phash_value}
3. Compare all pairs (i, j) where i < j:
   hamming = hash_i - hash_j  (imagehash overloads subtraction)
   if hamming <= threshold: record (filename_i, filename_j, hamming)
4. For each duplicate pair: mark the chronologically later file (by filename sort) as DUPLICATE_OF the earlier
OUTPUT: list of {filename, duplicate_of, hamming_distance}
```

#### Sequence Gap Analyser
```
INPUT: page_map = {filename: detected_page_no} (Nones excluded for sorting)
1. Sort by detected_page_no
2. Build sorted list of (filename, page_no) tuples
3. Iterate consecutive pairs (a, b):
   gap = b.page_no - a.page_no
   if gap == 1: no missing page
   if gap == 2: check alternating pattern flag
   if gap > 2: for each missing_no in range(a.page_no+1, b.page_no): add to MISSING list
4. Alternating pattern detection:
   if all gaps are exactly 2: pattern = 'alternating', suppress even/odd positions from MISSING
5. For images with page_no = None:
   - If bounded by known neighbors: status = INTERPOLATED
   - Else: status = UNKNOWN
OUTPUT: {missing_pages: [int], pattern: str, gaps: list, page_order: list}
```

---

### 5.3 Module 3 — OCR and NLP Extraction

**File:** `modules/ocr_extractor.py`

#### Tesseract Text Extractor
```
INPUT: image_path, lang='hin+san+eng'
1. Load and preprocess image (already done by preprocessor)
2. Run pytesseract.image_to_data(image, lang=lang, output_type=Output.DATAFRAME)
3. Filter rows: conf > -1 (Tesseract returns -1 for non-text regions)
4. Compute page_confidence = mean(filtered_conf_values) / 100.0
5. Concatenate all text values into raw_text string
OUTPUT: raw_text (str), page_confidence (float 0.0–1.0)
```

#### Script Detector
```
INPUT: text string
1. Count Devanagari chars: len([c for c in text if '\u0900' <= c <= '\u097F'])
2. Count Latin chars: len([c for c in text if c.isascii() and c.isalpha()])
3. total_alpha = devanagari_count + latin_count
4. if total_alpha == 0: return 'Unknown'
5. deva_ratio = devanagari_count / total_alpha
6. if deva_ratio > 0.6: return 'Devanagari'
7. if deva_ratio < 0.4: return 'Latin'
8. return 'Mixed'
```

#### NLP Text Cleaner
```
INPUT: raw OCR text string
1. Normalize Devanagari numerals (same function as sequence module)
2. Remove noise: strip characters that are not alphanumeric, space, or common punctuation
   re.sub(r'[^\w\s\u0900-\u097F.,;:!?\'"()-]', '', text)
3. Collapse multiple whitespace: re.sub(r'\s+', ' ', text).strip()
4. Remove standalone single characters that are not 'a', 'i', 'I' (OCR artifacts)
5. Return cleaned string
OUTPUT: cleaned_text (str)
```

#### Keyword Extractor
```
INPUT: cleaned_text (str), all_batch_texts (list of str), top_n=10
1. RAKE extraction:
   rake = Rake()
   rake.extract_keywords_from_text(cleaned_text)
   rake_keywords = rake.get_ranked_phrases_with_scores()  # list of (score, phrase)

2. TF-IDF re-ranking:
   vectorizer = TfidfVectorizer(max_features=200)
   vectorizer.fit(all_batch_texts)  # fit on entire batch
   tfidf_scores = vectorizer.transform([cleaned_text])
   tfidf_dict = dict(zip(vectorizer.get_feature_names_out(), tfidf_scores.toarray()[0]))

3. Combined score:
   for (rake_score, phrase) in rake_keywords:
       words = phrase.split()
       tfidf_boost = mean([tfidf_dict.get(w, 0) for w in words])
       combined = rake_score * 0.6 + tfidf_boost * 100 * 0.4

4. Return top_n by combined score
OUTPUT: list of (keyword_phrase, combined_score)
```

---

### 5.4 Aggregation Layer

**File:** `modules/aggregator.py`

#### Confidence Scorer
```
INPUT: merged_record dict, weights from config
Base weights: q=0.40, s=0.35, o=0.25

1. Identify which modules returned ERROR (score = -1 means error)
2. For each ERROR module: redistribute its weight equally to the non-error modules
3. final_score = quality_score*q_adj + sequence_confidence*s_adj + ocr_confidence*o_adj
4. Clamp to [0.0, 1.0]
OUTPUT: final_score (float)
```

#### Failure Router
```
INPUT: merged_record with final_score, review_threshold from config
1. if final_score >= review_threshold: route = 'results_ready'
2. else: route = 'human_review'
3. routing_reason = identify lowest-scoring module and its value
OUTPUT: route (str), routing_reason (str)
```

---

### 5.5 Output Layer

#### Dashboard Layout — `dashboard.py`
```
Sidebar:
  - Folder path input (text box)
  - Run pipeline button
  - Config override sliders (blur threshold, review threshold)

Main area row 1 — Metric cards (4 columns):
  - Total images processed
  - Images OK (green)
  - Images flagged (amber)
  - In human review (red)

Main area row 2 — Flagged gallery:
  - Grid of thumbnail images
  - Each thumbnail has: filename, status badge, final score

Main area row 3 — Full results table:
  - Pandas DataFrame with all CSV columns
  - Filter by status dropdown
  - Sort by any column

Main area row 4 — Downloads:
  - Download metadata CSV button
  - Download audit log JSON button
```

---

## 6. Error Handling Design

Every per-image function follows this pattern:

```python
def run_quality_check(image_path, config):
    try:
        image = load_image(image_path)
        # ... processing ...
        return result_dict
    except Exception as e:
        logger.error(f"Quality check failed for {image_path}: {e}")
        return {
            "quality_status": "ERROR",
            "quality_score": 0.0,
            "error_message": str(e)
        }
```

**Four non-negotiable rules:**
1. No single image failure crashes the batch — always wrap in try/except
2. Every error result has quality_score / sequence_confidence / ocr_confidence = 0.0
3. All error images route to human_review automatically
4. The audit log records the exception message for every ERROR image

---

## 7. Edge Cases and Handling Summary

| Edge Case | Where | Detection | Handling |
|-----------|-------|-----------|----------|
| Very dark image (not blurry) | Quality | mean_intensity < 30 | Flag DARK, not BLURRED |
| Decorative manuscript border | Quality | Contours touching image edge | Ignore in occlusion check |
| Diagonal Devanagari strokes | Quality | Short lines | Filter lines < 10% image width |
| Filename = `39(1907-1915).jpg` | Sequence | Bracket + year rules | Discard 39 and years, no page found |
| Only odd pages numbered | Sequence | Regular gap of 2 | Pattern = alternating, suppress blanks |
| Devanagari page numerals | Sequence | Unicode range check | Normalize to Arabic before analysis |
| Page number in body text | Sequence | Outlier + isolation filter | Body numbers fail isolation check |
| Faint/degraded page number | Sequence | Low Tesseract conf | Mark LOW_CONF, flag for review |
| Two identical scans | Sequence | pHash Hamming ≤ 8 | Mark DUPLICATE_OF, keep first |
| Completely unnumbered batch | Sequence | All layers return None | All pages = UNKNOWN, full review |
| OCR fails on Sarada script | OCR | Low page_confidence | Flag LOW_CONF, human review |
| Colophon image is blurry | OCR | Cross-reference quality | Note in extracted_text: "low quality source" |

---

## 8. requirements.txt

```
opencv-python>=4.7.0
Pillow>=9.0.0
pytesseract>=0.3.10
imagehash>=4.3.1
spacy>=3.5.0
rake-nltk>=1.0.6
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
streamlit>=1.20.0
PyYAML>=6.0
tqdm>=4.0.0
```

**After pip install, also run:**
```
# Install Tesseract engine (Ubuntu/Debian)
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-hin

# Download spaCy English model
python -m spacy download en_core_web_sm
```