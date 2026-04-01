# IDP for Heritage Manuscript Archives
## Task Breakdown Document

**Project:** Intelligent Document Processing (IDP) for Heritage Manuscript Archives
**Team:** CSE – Data Science, VIT
**Build Order:** Module by module, one at a time, easiest → hardest

---

## Implementation Order Overview

```
Phase 0 → Project Setup & Config
Phase 1 → Preprocessing Pipeline
Phase 2 → Module 1: Image Quality Checker     ← Start here (easiest)
Phase 3 → Module 2: Sequence Detection        ← Middle complexity
Phase 4 → Module 3: OCR + NLP Extraction      ← Needs setup time
Phase 5 → Aggregation Layer
Phase 6 → Output: Dashboard + CSV + Audit Log
Phase 7 → Testing, Evaluation, Final Report
```

---

## Phase 0 — Project Setup

### TASK-0.1: Repository and folder structure
- Create project folder: `idp_manuscript/`
- Create subfolders: `modules/`, `data/sample_images/`, `outputs/`, `models/`, `tests/`, `docs/`
- Initialize git repository
- Create `.gitignore` (exclude large image files, model weights, output CSVs)
- **Output:** Working repo with folder structure

### TASK-0.2: Config file
- Create `config.yaml` with all threshold values
- Include: blur_threshold (100), skew_max_angle (3.0), occlusion_area_pct (5.0), phash_threshold (8), ocr_confidence_min (0.6), review_threshold (0.5), resize_max_px (2000), languages ('hin+san+eng')
- **Output:** `config.yaml`

### TASK-0.3: Dependencies and environment
- Create `requirements.txt` with all libraries
- Install and verify: opencv-python, Pillow, pytesseract, imagehash, spacy, rake-nltk, scikit-learn, pandas, streamlit, PyYAML, tqdm
- Install Tesseract engine + Hindi (hin) language pack
- Download spaCy model: `python -m spacy download en_core_web_sm`
- **Output:** `requirements.txt`, working virtual environment

### TASK-0.4: Logger and base utilities
- Create `utils/logger.py` — unified logging to file + console
- Create `utils/file_handler.py` — image loader, batch walker, output writer
- Create `utils/image_utils.py` — shared helpers (load image, save image, resize)
- **Output:** `utils/` folder with 3 files

---

## Phase 1 — Preprocessing Pipeline

### TASK-1.1: Image loader and validator
- Function: `load_batch(folder_path)` → returns list of image paths
- Validate: file exists, readable, is a valid image, format is JPG/PNG/TIFF
- Log: total images found, any skipped files
- **Output:** `modules/preprocessor.py` (partial)

### TASK-1.2: Preprocessing transforms
- Function: `preprocess_image(image_path, config)` → returns processed numpy array
- Steps: load → grayscale → resize (preserve aspect ratio) → Gaussian denoise → normalize
- Keep original untouched; work on copy
- **Output:** `modules/preprocessor.py` (complete)

### TASK-1.3: Preprocessing test
- Test on 5 sample images: print shape before and after, display side-by-side comparison
- Verify grayscale conversion, no dimension errors on different aspect ratios
- **Output:** `tests/test_preprocessor.py`

---

## Phase 2 — Module 1: Image Quality Checker

### TASK-2.1: Blur detector
- Function: `detect_blur(gray_image, threshold)` → returns (is_blurred: bool, score: float, variance: float)
- Method: `cv2.Laplacian(image, cv2.CV_64F).var()`
- Score formula: `min(variance / threshold, 1.0)`
- Flag: BLURRED if variance < threshold
- **Output:** `modules/quality_checker.py` (partial)

### TASK-2.2: Skew detector
- Function: `detect_skew(gray_image, max_angle)` → returns (is_skewed: bool, angle: float)
- Method: Canny edge detection → HoughLinesP → compute median angle of detected lines
- Flag: SKEWED if abs(angle) > max_angle
- **Output:** `modules/quality_checker.py` (partial)

### TASK-2.3: Crop detector
- Function: `detect_crop(gray_image)` → returns (is_cropped: bool, edges_affected: list)
- Method: check pixel intensity along all 4 borders; if any border row/column has mean intensity below threshold, flag as cropped edge
- Return which edges are affected: ['top', 'bottom', 'left', 'right']
- **Output:** `modules/quality_checker.py` (partial)

### TASK-2.4: Occlusion detector
- Function: `detect_occlusion(gray_image, area_threshold_pct)` → returns (is_occluded: bool, area_pct: float)
- Method: threshold image for dark blobs → find contours → check if any contour covers > area_threshold_pct% of image
- Flag irregular dark shapes that are not part of manuscript text patterns
- **Output:** `modules/quality_checker.py` (partial)

### TASK-2.5: Quality aggregator function
- Function: `run_quality_check(image_path, config)` → returns dict with all quality results
- Combine all 4 detectors; compute final quality_score (weighted)
- Assign status: OK / BLURRED / SKEWED / CROPPED / OCCLUDED / MULTI_FLAG / ERROR
- Return: {filename, quality_status, quality_score, blur_variance, skew_angle, is_cropped, is_occluded}
- **Output:** `modules/quality_checker.py` (complete)

### TASK-2.6: Quality module test
- Test on: 1 known blurry image, 1 known tilted image, 1 clean image
- Print status and score for each; verify correct flagging
- **Output:** `tests/test_quality.py`

---

## Phase 3 — Module 2: Sequence and Duplicate Detection

### TASK-3.1: Filename parser (Layer 1)
- Function: `parse_filename(filename)` → returns (page_number: int or None, confidence: float)
- Use regex to extract all numeric patterns
- Filter: discard 4-digit numbers 1600–2025 (years), discard numbers immediately before `(` or `[` (manuscript IDs)
- Confidence: 0.9 if a clean number found, 0.0 if nothing remains after filtering
- **Output:** `modules/sequence_detector.py` (partial)

### TASK-3.2: Region OCR — Layer 2
- Function: `ocr_region(image_path, region)` → returns (page_number: int or None, confidence: float)
- Crop top 20% and bottom 20% of image
- Run pytesseract with `--psm 7` (single line) on each crop
- Extract isolated numeric tokens; return best candidate with confidence
- Normalize Devanagari numerals to Arabic before returning
- **Output:** `modules/sequence_detector.py` (partial)

### TASK-3.3: Full-page OCR with outlier filter — Layer 3
- Function: `ocr_full_page(image_path, batch_y_cluster)` → returns (page_number: int or None, confidence: float)
- Run pytesseract with `image_to_data` to get word-level bounding boxes and text
- Extract all numeric tokens with their y-coordinates
- Use pre-computed batch y-cluster (most common y-band for page numbers across the batch) to score candidates
- Short isolated numbers in the cluster band score highest
- Normalize Devanagari numerals
- **Output:** `modules/sequence_detector.py` (partial)

### TASK-3.4: Devanagari numeral normalizer
- Function: `normalize_numerals(text)` → returns string with Devanagari digits replaced by Arabic digits
- Lookup dict: {'०':'0','१':'1','२':'2','३':'3','४':'4','५':'5','६':'6','७':'7','८':'8','९':'9'}
- Apply before all numeric extraction
- **Output:** Added to `modules/sequence_detector.py`

### TASK-3.5: Page number extractor (layer orchestrator)
- Function: `extract_page_number(image_path, filename, batch_y_cluster, config)` → returns (page_number, confidence, method_used)
- Try Layer 1; if confidence > 0.7, return
- Else try Layer 2; if confidence > 0.7, return
- Else try Layer 3; return whatever Layer 3 gives
- Record method_used: 'filename' / 'region_ocr' / 'full_ocr' / 'unknown'
- **Output:** `modules/sequence_detector.py` (partial)

### TASK-3.6: pHash duplicate detector
- Function: `detect_duplicates(image_paths, threshold)` → returns list of (img_a, img_b, hamming_distance)
- Compute pHash for every image using imagehash library
- Compare all pairs; flag pairs with Hamming distance ≤ threshold
- Return structured list; mark the later scan as DUPLICATE_OF the earlier
- **Output:** `modules/sequence_detector.py` (partial)

### TASK-3.7: Sequence gap analyser
- Function: `analyze_sequence(page_map)` → returns (missing_pages: list, pattern: str, gaps: list)
- Sort images by detected page number
- Detect gaps: any jump > 1 between consecutive detected numbers
- Detect alternating pattern: if gaps appear at regular intervals (every 2nd), set pattern='alternating', suppress expected blanks from MISSING
- Classify: MISSING / EXPECTED_BLANK / INTERPOLATED / UNKNOWN for each unaccounted position
- **Output:** `modules/sequence_detector.py` (complete)

### TASK-3.8: Sequence module test
- Test filename parser on: `39(1907-1915).jpg`, `page_05.jpg`, `MS_042_recto.jpg`, `001.jpg`
- Test gap analyser on: [1,2,4,5,8] → should detect gaps at 3 and 6,7
- Test duplicate detector on 2 identical images and 2 different images
- **Output:** `tests/test_sequence.py`

---

## Phase 4 — Module 3: OCR and NLP Extraction

### TASK-4.1: Tesseract setup and text extractor
- Function: `extract_text(image_path, lang)` → returns (raw_text: str, word_confidences: list, page_confidence: float)
- Run `pytesseract.image_to_data()` with `lang='hin+san+eng'`
- Compute page_confidence as mean of word-level confidence scores (conf > -1 only)
- Return raw OCR output plus confidence
- **Output:** `modules/ocr_extractor.py` (partial)

### TASK-4.2: Script detector
- Function: `detect_script(text)` → returns 'Devanagari' / 'Latin' / 'Mixed' / 'Unknown'
- Count characters in Unicode ranges: Devanagari (U+0900–U+097F), Latin (A-z)
- Whichever exceeds 60% of total non-space characters → that script
- Both below 60% → Mixed
- **Output:** `modules/ocr_extractor.py` (partial)

### TASK-4.3: NLP text cleaner
- Function: `clean_text(raw_text)` → returns cleaned string
- Steps: remove noise characters, normalize whitespace, strip non-linguistic symbols, handle OCR ligature errors
- Use regex for noise removal; spaCy for tokenization
- **Output:** `modules/ocr_extractor.py` (partial)

### TASK-4.4: Keyword extractor
- Function: `extract_keywords(cleaned_text, top_n)` → returns list of (keyword, score) tuples
- Primary: RAKE-NLTK to generate candidate keywords
- Re-rank using TF-IDF across the batch (fit TF-IDF on all page texts, use score to filter)
- Return top_n=10 keywords with scores
- **Output:** `modules/ocr_extractor.py` (partial)

### TASK-4.5: Colophon and title extractor
- Function: `extract_manuscript_summary(batch_image_paths, config)` → returns dict
- Apply full OCR + NLP to first image (title folio) and last image (Colophon) of the batch
- Return: {title_text, title_keywords, colophon_text, colophon_keywords, detected_script}
- **Output:** `modules/ocr_extractor.py` (complete)

### TASK-4.6: OCR module test
- Test on 1 Devanagari image and 1 Latin/mixed image
- Print raw text, confidence, detected script, top 5 keywords
- Verify Devanagari numeral normalization works within OCR pipeline
- **Output:** `tests/test_ocr.py`

---

## Phase 5 — Aggregation Layer

### TASK-5.1: Result merger
- Function: `merge_results(quality_result, sequence_result, ocr_result)` → returns unified record dict
- Combine all module outputs into one dict per image
- Handle missing module results (ERROR state) by redistributing weights
- **Output:** `modules/aggregator.py` (partial)

### TASK-5.2: Confidence scorer
- Function: `score_confidence(merged_record)` → returns final_score (float 0.0–1.0)
- Weighted formula: quality_score × 0.40 + sequence_confidence × 0.35 + ocr_confidence × 0.25
- If a module returned ERROR: redistribute its weight equally to the others
- **Output:** `modules/aggregator.py` (partial)

### TASK-5.3: Failure router
- Function: `route_record(merged_record, threshold)` → returns 'results_ready' or 'human_review'
- If final_score < threshold (default 0.5) → human_review
- If final_score ≥ threshold → results_ready
- Attach routing_reason (which module caused low score)
- **Output:** `modules/aggregator.py` (complete)

### TASK-5.4: Aggregation test
- Test merger on 3 mock records: one fully passing, one with quality ERROR, one with low OCR score
- Verify routing decisions are correct
- **Output:** `tests/test_aggregator.py`

---

## Phase 6 — Output Layer

### TASK-6.1: CSV exporter
- Function: `export_csv(records, output_path)` → writes metadata CSV
- Columns: filename, detected_page_no, detection_method, quality_status, quality_score, sequence_status, ocr_confidence, detected_script, top_keywords, final_score, pipeline_status, notes
- Use pandas; sort by detected_page_no
- **Output:** `modules/exporter.py` (partial)

### TASK-6.2: Audit log writer
- Function: `write_audit_log(review_queue, output_path)` → writes JSON
- Each entry: full per-module details, routing reason, timestamp
- Formatted JSON, human-readable
- **Output:** `modules/exporter.py` (complete)

### TASK-6.3: Streamlit dashboard
- Page layout: sidebar (folder upload, config controls) + main area (summary metrics, flagged image gallery, full data table)
- Summary metrics: total images, % OK, % flagged by type, % in human review
- Flagged gallery: thumbnails with status badge and score
- Full table: sortable, filterable pandas dataframe display
- Download buttons: CSV and audit log JSON
- **Output:** `dashboard.py`

### TASK-6.4: Main pipeline runner
- File: `main.py`
- Orchestrate all stages in order: input → preprocess → quality → sequence → ocr → aggregate → export
- Accept CLI argument: `python main.py --input ./data/batch1 --config config.yaml --output ./outputs/`
- Log start time, end time, total processed
- **Output:** `main.py`

---

## Phase 7 — Testing and Evaluation

### TASK-7.1: End-to-end test on sample batch
- Run full pipeline on 20–30 sample images
- Verify CSV is produced, audit log is written, dashboard launches
- Check that all statuses are present in output

### TASK-7.2: Evaluation metrics
- Blur detection: precision and recall on labelled test set
- Sequence detection: % of page numbers correctly identified
- OCR: character error rate (CER) on known ground truth
- Overall: % of images correctly classified (OK vs flagged)

### TASK-7.3: Performance test
- Time the pipeline on batches of 50, 100, 200 images
- Identify bottleneck (expected: OCR module)
- Optimize if batch of 200 exceeds 15 minutes

### TASK-7.4: Report and documentation
- Write method section for each module
- Document all config parameters
- Create README.md with setup and run instructions
- Record demo video

---

## Summary: Files to Create

```
idp_manuscript/
├── config.yaml                  TASK-0.2
├── requirements.txt             TASK-0.3
├── main.py                      TASK-6.4
├── dashboard.py                 TASK-6.3
├── utils/
│   ├── logger.py                TASK-0.4
│   ├── file_handler.py          TASK-0.4
│   └── image_utils.py           TASK-0.4
├── modules/
│   ├── preprocessor.py          TASK-1.1 + 1.2
│   ├── quality_checker.py       TASK-2.1 to 2.5
│   ├── sequence_detector.py     TASK-3.1 to 3.7
│   ├── ocr_extractor.py         TASK-4.1 to 4.5
│   ├── aggregator.py            TASK-5.1 to 5.3
│   └── exporter.py              TASK-6.1 + 6.2
├── tests/
│   ├── test_preprocessor.py     TASK-1.3
│   ├── test_quality.py          TASK-2.6
│   ├── test_sequence.py         TASK-3.8
│   ├── test_ocr.py              TASK-4.6
│   └── test_aggregator.py       TASK-5.4
├── data/
│   └── sample_images/
└── outputs/
    ├── metadata.csv
    └── audit_log.json
```