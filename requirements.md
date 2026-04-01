# IDP for Heritage Manuscript Archives
## Requirements Document

**Project:** Intelligent Document Processing (IDP) for Heritage Manuscript Archives
**Sponsor:** Heritage Conservation Organization
**Team:** CSE – Data Science, VIT
**Version:** 1.0

---

## 1. Project Overview

Develop an automated Python pipeline that processes batches of high-resolution manuscript scan images and produces a quality audit report, a metadata CSV index, and a Streamlit dashboard. The system replaces a fully manual post-digitization QC workflow for Sanskrit and Marathi heritage manuscripts written in Devanagari and Sarada scripts.

The pipeline has 7 stages as per architecture: Input → Preprocessing → Quality Module → Sequence Module → OCR Module → Aggregation → Output, with a Human Review fallback for low-confidence results.

---

## 2. Functional Requirements

### 2.1 Stage 1 — Input Layer

| ID | Requirement |
|----|-------------|
| FR-1.1 | System shall accept a folder path containing manuscript images in JPG, PNG, and TIFF formats. |
| FR-1.2 | System shall load a YAML configuration file containing all threshold values, language settings, and pipeline switches before processing begins. |
| FR-1.3 | System shall validate that the input folder is non-empty and that all files are readable before starting the batch. |
| FR-1.4 | System shall produce an "Inputs Ready" signal only after both image loading and config loading succeed. |
| FR-1.5 | System shall log the total image count, formats detected, and config values loaded at the start of every run. |

### 2.2 Stage 2 — Preprocessing

| ID | Requirement |
|----|-------------|
| FR-2.1 | System shall convert all images to grayscale for uniform downstream processing. |
| FR-2.2 | System shall resize all images to a standard resolution defined in config (default: longest side = 2000px, preserving aspect ratio). |
| FR-2.3 | System shall apply Gaussian denoising to reduce scan artifacts before feature extraction. |
| FR-2.4 | System shall normalize pixel intensity values to a 0–255 range. |
| FR-2.5 | System shall retain the original high-resolution copy; preprocessing shall operate on a working copy only. |
| FR-2.6 | System shall produce an "Images Normalized" signal when preprocessing is complete for the entire batch. |

### 2.3 Stage 3 — Quality Module (Module 1)

| ID | Requirement |
|----|-------------|
| FR-3.1 | System shall detect blurred or out-of-focus images using Laplacian variance. Images with variance below a configurable threshold (default: 100) shall be flagged BLURRED. |
| FR-3.2 | System shall compute a blur confidence score: score = min(laplacian_variance / threshold, 1.0). |
| FR-3.3 | System shall detect skewed manuscript folios using Hough Line Transform. Images with detected skew angle beyond ±3° shall be flagged SKEWED. |
| FR-3.4 | System shall detect aggressive or incorrect cropping where manuscript content appears cut off at any image edge. |
| FR-3.5 | System shall detect partial occlusions — dark irregular blobs (fingers, tools, shadows) — using contour analysis. A region covering more than 5% of the image area that is significantly darker than the median shall be flagged OCCLUDED. |
| FR-3.6 | Each image shall receive an overall quality confidence score between 0.0 (worst) and 1.0 (best) combining all sub-checks. |
| FR-3.7 | Each image shall be assigned a quality status: OK, BLURRED, SKEWED, CROPPED, OCCLUDED, or MULTI_FLAG (more than one issue). |
| FR-3.8 | A "Sequence Check" cross-reference signal shall be passed from the Quality Module to the Sequence Module for images flagged CROPPED, as cropping may have removed page number regions. |
| FR-3.9 | System shall never crash the batch on a single image failure; failed images shall be assigned status ERROR with confidence 0.0 and processing shall continue. |

### 2.4 Stage 4 — Sequence Module (Module 2)

| ID | Requirement |
|----|-------------|
| FR-4.1 | System shall attempt page number extraction using three layers in order, stopping when confidence exceeds 0.7. |
| FR-4.2 | Layer 1 — Filename parsing: System shall extract candidate numbers from filenames using regex and shall discard 4-digit numbers in the range 1600–2025 (years) and numbers appearing immediately before a bracket (manuscript IDs). |
| FR-4.3 | Layer 2 — Region OCR: System shall crop the top 20% and bottom 20% of each image and run Tesseract OCR to find margin page numbers. |
| FR-4.4 | Layer 3 — Full-page OCR with outlier filter: System shall run Tesseract on the full image, extract all numeric tokens, and apply position clustering across the batch to identify the y-coordinate band where page numbers consistently appear. |
| FR-4.5 | System shall normalize Devanagari numerals (०–९) to Arabic numerals (0–9) before all sequence analysis. |
| FR-4.6 | System shall detect duplicate scans using perceptual hashing (pHash) with a Hamming distance threshold of ≤ 8 (configurable). Duplicates shall be flagged with a DUPLICATE_OF reference to the original filename. |
| FR-4.7 | System shall sort the batch by detected page number and perform gap analysis. Each gap in the sequence shall be reported as a MISSING entry with the expected page number. |
| FR-4.8 | System shall detect alternating-page numbering patterns (e.g., only odd pages numbered). Detected blanks at regular intervals shall be reported as EXPECTED_BLANK, not MISSING. |
| FR-4.9 | Pages where no number can be determined shall be assigned status INTERPOLATED if bounded by known neighbors, or UNKNOWN otherwise. |
| FR-4.10 | A "Flags" signal with all MISSING, DUPLICATE, and UNKNOWN entries shall be passed to the Aggregation stage. |

### 2.5 Stage 5 — OCR Module (Module 3)

| ID | Requirement |
|----|-------------|
| FR-5.1 | System shall run Tesseract OCR on each image with language configuration: `hin+san+eng`. |
| FR-5.2 | System shall extract text specifically from the first page (title folio) and last page (Colophon) of each manuscript batch. |
| FR-5.3 | System shall clean OCR output using NLP: strip noise characters, normalize whitespace, remove non-linguistic symbols, and handle OCR ligature errors common in Devanagari. |
| FR-5.4 | System shall extract the top 10 keywords per manuscript using RAKE. TF-IDF shall be used as a re-ranking step across the batch. |
| FR-5.5 | System shall compute a page-level OCR confidence score aggregated from Tesseract word-level confidence values. |
| FR-5.6 | System shall identify which script is present (Devanagari / Latin / Mixed) and record it in the output. |
| FR-5.7 | A "Keywords" and "Gaps" signal shall be passed from the OCR Module to the Aggregation stage. |

### 2.6 Stage 6 — Aggregation

| ID | Requirement |
|----|-------------|
| FR-6.1 | System shall merge results from all three modules into a single record per image. |
| FR-6.2 | System shall compute a final confidence score per image as a weighted average: quality (40%), sequence (35%), OCR (25%). |
| FR-6.3 | System shall route images with final confidence below 0.5 to the Human Review queue. |
| FR-6.4 | System shall route images with final confidence ≥ 0.5 to the Results Ready pool. |
| FR-6.5 | The aggregator shall never block on a module failure. If a module returned ERROR for an image, that module's weight is redistributed to the others. |

### 2.7 Stage 7 — Output Layer

| ID | Requirement |
|----|-------------|
| FR-7.1 | System shall generate a Streamlit dashboard showing: total images processed, count by status, flagged image thumbnails, per-image details, and downloadable CSV. |
| FR-7.2 | System shall export a metadata CSV with columns: filename, detected_page_no, detection_method, quality_status, quality_score, sequence_status, ocr_confidence, detected_script, keywords, final_score, status, notes. |
| FR-7.3 | System shall write a JSON audit log containing all Human Review queue entries with full per-module details for manual resolution. |
| FR-7.4 | On manual resolution (Human Review → Resolved), the resolved record shall re-enter the Results Ready pool and the pipeline shall be marked complete. |
| FR-7.5 | System shall produce a "Pipeline Complete" signal only after all images have a final status (either automated or manually resolved). |

---

## 3. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-1 | The pipeline shall process a batch of 500 images in under 30 minutes on a standard laptop (8GB RAM, no GPU required for core modules). |
| NFR-2 | All module functions shall be individually testable with a single image input. |
| NFR-3 | All thresholds shall be configurable via config.yaml without changing source code. |
| NFR-4 | The system shall support Python 3.9 and above. |
| NFR-5 | All dependencies shall be installable via a single `pip install -r requirements.txt` command. |
| NFR-6 | The codebase shall be modular: each module (quality, sequence, ocr, aggregator, dashboard) shall be a separate Python file importable independently. |
| NFR-7 | The system shall not require internet access during processing (all models and language packs installed locally). |

---

## 4. Dataset Requirements

| Dataset | Purpose | Source |
|---------|---------|--------|
| CERTH Image Blur Dataset | Blur detection training/testing | Kaggle |
| Devanagari Handwritten Character Dataset | OCR testing | Kaggle (92,000 images) |
| IIIT-HW-Dev | Handwritten Devanagari text lines | IIT Bombay |
| BHOOMI Dataset | Indian land records in Devanagari | Government open data |
| Palm Leaf Manuscript Dataset | Actual manuscript images | Zenodo |
| RVL-CDIP (subset) | Document classification backbone | HuggingFace / Kaggle |
| Custom sponsor samples | Real-world validation | Provided by sponsor |

---

## 5. Deliverables

1. Python pipeline codebase (modular, version-controlled)
2. Streamlit dashboard (uploadable folder, visual audit report)
3. Metadata CSV template and populated sample output
4. Trained lightweight CNN or rule-based model for quality detection
5. Final project report with methodology, results, and evaluation metrics
6. Demo video of the pipeline running on a real manuscript batch