"""
Test runner script that writes output to a file (avoids Windows console encoding issues).
"""
import os
import sys

# Force UTF-8 and redirect to file
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results.txt")
log_file = open(log_path, "w", encoding="utf-8")

class DualWriter:
    def __init__(self, file):
        self.file = file
        self.orig = sys.stdout
    def write(self, text):
        self.file.write(text)
        try:
            self.orig.write(text)
        except Exception:
            pass
    def flush(self):
        self.file.flush()
        try:
            self.orig.flush()
        except Exception:
            pass

sys.stdout = DualWriter(log_file)
sys.stderr = DualWriter(log_file)

# Suppress pytesseract warnings
import logging
logging.disable(logging.WARNING)

# Now add project root and import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2


def create_test_image(width=400, height=500, add_text=True, blur=False):
    img = np.ones((height, width, 3), dtype=np.uint8) * 230
    cv2.rectangle(img, (20, 20), (width-20, height-20), (100, 80, 60), 2)
    cv2.line(img, (30, 60), (width-30, 60), (100, 80, 60), 1)
    if add_text:
        cv2.putText(img, "Test Manuscript Page", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
        cv2.putText(img, "Sample text line one", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
        cv2.putText(img, "Sample text line two", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    if blur:
        img = cv2.GaussianBlur(img, (21, 21), 10)
    return img


passed = 0
failed = 0

# ----------------------------------------------------------------
# TEST 1: Preprocessor
# ----------------------------------------------------------------
print("=" * 60)
print("TEST 1: Preprocessor Module")
print("=" * 60)
try:
    from modules.preprocessor import preprocess_image
    img = create_test_image()
    result = preprocess_image(img)
    assert result["success"], "Preprocessing should succeed"
    assert result["processed"] is not None
    assert "grayscale" in result["steps"]
    assert "denoised" in result["steps"]
    assert "enhanced" in result["steps"]
    print(f"  Original size:  {result['original_size']}")
    print(f"  Processed size: {result['processed_size']}")
    print(f"  Steps: {list(result['steps'].keys())}")
    print("  >> PASSED\n")
    passed += 1
except Exception as e:
    print(f"  >> FAILED: {e}\n")
    failed += 1

# ----------------------------------------------------------------
# TEST 2: Quality Checker
# ----------------------------------------------------------------
print("=" * 60)
print("TEST 2: Quality Checker Module")
print("=" * 60)
try:
    from modules.quality_checker import check_blur, check_skew, run_quality_check

    sharp = create_test_image(blur=False)
    blur_sharp = check_blur(sharp)
    print(f"  Sharp image blur score: {blur_sharp['blur_score']}")

    blurry = create_test_image(blur=True)
    blur_blurry = check_blur(blurry)
    print(f"  Blurry image blur score: {blur_blurry['blur_score']}")

    assert blur_sharp["blur_score"] > blur_blurry["blur_score"], "Sharp should score higher"
    assert blur_blurry["is_blurred"] == True, "Blurry image should be flagged"

    skew_r = check_skew(sharp)
    print(f"  Skew angle: {skew_r['skew_angle']} deg")

    quality = run_quality_check(sharp)
    print(f"  Quality score: {quality['quality_score']}/100")
    print(f"  Overall status: {quality['overall_status']}")
    assert quality["quality_score"] > 0
    print("  >> PASSED\n")
    passed += 1
except Exception as e:
    print(f"  >> FAILED: {e}\n")
    failed += 1

# ----------------------------------------------------------------
# TEST 3: Sequence Detector
# ----------------------------------------------------------------
print("=" * 60)
print("TEST 3: Sequence Detector Module")
print("=" * 60)
try:
    from modules.sequence_detector import run_sequence_detection
    filenames = [
        "manuscript_page_001.png",
        "manuscript_page_002.png",
        "manuscript_page_003.png",
        "manuscript_page_005.png",
        "manuscript_page_006.png",
    ]
    results = run_sequence_detection(filenames)
    for r in results["per_file"]:
        print(f"  {r['filename']} -> Page {r['page_number']} ({r['method']})")
    gaps = results["gap_analysis"]
    print(f"  Missing pages: {gaps['missing_pages']}")
    assert 4 in gaps["missing_pages"], "Should detect page 4 missing"
    print("  >> PASSED\n")
    passed += 1
except Exception as e:
    print(f"  >> FAILED: {e}\n")
    failed += 1

# ----------------------------------------------------------------
# TEST 4: OCR Engine (graceful without Tesseract)
# ----------------------------------------------------------------
print("=" * 60)
print("TEST 4: OCR Engine Module")
print("=" * 60)
try:
    from modules.ocr_engine import run_ocr, extract_keywords, detect_script

    img = create_test_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = run_ocr(gray)
    print(f"  OCR success: {result['success']}")
    if not result['success']:
        print(f"  (Tesseract not installed - fallback mode OK)")

    kw = extract_keywords("heritage manuscript ancient calligraphy heritage text")
    print(f"  Keywords: {kw}")
    assert len(kw) > 0

    sc = detect_script("Hello World this is English text for testing purposes")
    print(f"  Script detected: {sc}")
    assert sc["script"] == "Latin"
    print("  >> PASSED\n")
    passed += 1
except Exception as e:
    print(f"  >> FAILED: {e}\n")
    failed += 1

# ----------------------------------------------------------------
# TEST 5: Aggregator
# ----------------------------------------------------------------
print("=" * 60)
print("TEST 5: Aggregator Module")
print("=" * 60)
try:
    from modules.aggregator import aggregate_results, generate_batch_summary
    prep = [{"filename": "test_001.png"}, {"filename": "test_002.png"}]
    quality = [
        {"quality_score": 85, "blur": {"blur_status": "OK"}, "skew": {"skew_status": "OK"}},
        {"quality_score": 40, "blur": {"blur_status": "BLURRED"}, "skew": {"skew_status": "OK"}},
    ]
    sequence = {"per_file": [{"page_number": 1}, {"page_number": 2}], "gap_analysis": {"missing_pages": []}}
    ocr = [{"confidence": 78.5}, {"confidence": 45.0}]
    agg = aggregate_results(prep, quality, sequence, ocr)
    summary = generate_batch_summary(agg)
    for r in agg:
        print(f"  {r['filename']}: final={r['final_score']}, status={r['status']}")
    print(f"  Batch: total={summary['total_images']}, ready={summary['ready']}, pass_rate={summary['pass_rate']}%")
    assert len(agg) == 2
    print("  >> PASSED\n")
    passed += 1
except Exception as e:
    print(f"  >> FAILED: {e}\n")
    failed += 1

# ----------------------------------------------------------------
# TEST 6: Output Generator
# ----------------------------------------------------------------
print("=" * 60)
print("TEST 6: Output Generator Module")
print("=" * 60)
try:
    import json
    from modules.output_generator import generate_csv, generate_json_audit
    agg_data = [
        {"filename": "test_001.png", "quality_score": 85, "page_number": 1,
         "ocr_confidence": 78.5, "final_score": 85.4, "status": "READY",
         "blur_status": "OK", "skew_status": "OK"},
    ]
    summary_data = {"total_images": 1, "ready": 1, "review": 0, "rejected": 0}
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "test_run")
    csv_path = generate_csv(agg_data, os.path.join(out_dir, "test.csv"))
    json_path = generate_json_audit(agg_data, summary_data, os.path.join(out_dir, "test.json"))
    assert csv_path and os.path.exists(csv_path)
    assert json_path and os.path.exists(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        audit = json.load(f)
    assert audit["metadata"]["system"] == "IDP Manuscript Processing System"
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    print("  >> PASSED\n")
    passed += 1
except Exception as e:
    print(f"  >> FAILED: {e}\n")
    failed += 1

# ----------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------
print("=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed, {passed+failed} total")
if failed == 0:
    print("ALL MODULE TESTS PASSED!")
else:
    print(f"WARNING: {failed} test(s) failed")
print("=" * 60)

log_file.close()
sys.exit(failed)
