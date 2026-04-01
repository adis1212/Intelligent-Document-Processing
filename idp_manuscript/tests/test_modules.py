"""
Tests for IDP Manuscript Processing Modules
Handles Windows console encoding gracefully.
"""

import os
import sys
import io

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import cv2

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_test_image(width=400, height=500, add_text=True, blur=False):
    """Create a synthetic test image."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 230
    cv2.rectangle(img, (20, 20), (width - 20, height - 20), (100, 80, 60), 2)
    cv2.line(img, (30, 60), (width - 30, 60), (100, 80, 60), 1)
    if add_text:
        cv2.putText(img, "Test Manuscript Page", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
        cv2.putText(img, "Sample text line 1", (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
        cv2.putText(img, "Sample text line 2", (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    if blur:
        img = cv2.GaussianBlur(img, (21, 21), 10)
    return img


def test_preprocessor():
    """Test the preprocessor module."""
    print("\n" + "=" * 50)
    print("TEST: Preprocessor Module")
    print("=" * 50)

    from modules.preprocessor import preprocess_image

    img = create_test_image()
    result = preprocess_image(img)

    assert result["success"], "Preprocessing should succeed"
    assert result["processed"] is not None, "Should produce processed image"
    assert "grayscale" in result["steps"], "Should include grayscale step"
    assert "denoised" in result["steps"], "Should include denoised step"
    assert "enhanced" in result["steps"], "Should include enhanced step"

    print(f"  [OK] Original size:  {result['original_size']}")
    print(f"  [OK] Processed size: {result['processed_size']}")
    print(f"  [OK] Steps: {list(result['steps'].keys())}")
    print("  [PASS] Preprocessor Module -- ALL TESTS PASSED")
    return True


def test_quality_checker():
    """Test the quality checker module."""
    print("\n" + "=" * 50)
    print("TEST: Quality Checker Module")
    print("=" * 50)

    from modules.quality_checker import check_blur, check_skew, run_quality_check

    # Sharp image
    sharp = create_test_image(blur=False)
    blur_result = check_blur(sharp)
    print(f"  Sharp image blur score: {blur_result['blur_score']}")
    print(f"  Is blurred: {blur_result['is_blurred']}")

    # Blurry image
    blurry = create_test_image(blur=True)
    blur_result_blurred = check_blur(blurry)
    print(f"  Blurry image blur score: {blur_result_blurred['blur_score']}")
    print(f"  Is blurred: {blur_result_blurred['is_blurred']}")

    assert blur_result["blur_score"] > blur_result_blurred["blur_score"], \
        "Sharp image should have higher blur score"

    # Skew test
    skew_result = check_skew(sharp)
    print(f"  Skew angle: {skew_result['skew_angle']} degrees")

    # Full quality check
    quality = run_quality_check(sharp)
    print(f"  Quality score: {quality['quality_score']}")
    print(f"  Overall status: {quality['overall_status']}")
    assert quality["quality_score"] > 0, "Quality score should be positive"

    print("  [PASS] Quality Checker Module -- ALL TESTS PASSED")
    return True


def test_sequence_detector():
    """Test the sequence detector module."""
    print("\n" + "=" * 50)
    print("TEST: Sequence Detector Module")
    print("=" * 50)

    from modules.sequence_detector import run_sequence_detection

    filenames = [
        "manuscript_page_001.png",
        "manuscript_page_002.png",
        "manuscript_page_003.png",
        "manuscript_page_005.png",  # Gap at page 4
        "manuscript_page_006.png",
    ]

    results = run_sequence_detection(filenames)

    for r in results["per_file"]:
        print(f"  {r['filename']} -> Page {r['page_number']} ({r['method']})")

    gaps = results["gap_analysis"]
    print(f"  Missing pages: {gaps['missing_pages']}")
    print(f"  Range: {gaps['range']}")

    assert 4 in gaps["missing_pages"], "Should detect page 4 is missing"
    print("  [PASS] Sequence Detector Module -- ALL TESTS PASSED")
    return True


def test_ocr_engine():
    """Test the OCR engine module (graceful fallback if tesseract missing)."""
    print("\n" + "=" * 50)
    print("TEST: OCR Engine Module")
    print("=" * 50)

    from modules.ocr_engine import run_ocr, extract_keywords, detect_script

    img = create_test_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result = run_ocr(gray)
    print(f"  OCR success: {result['success']}")
    if result['success']:
        print(f"  Text extracted: {result['text'][:80]}...")
        print(f"  Confidence: {result['confidence']}%")
    else:
        print(f"  [INFO] Tesseract not installed - fallback mode active (this is OK)")
        print(f"  Fallback text: {result['text'][:80]}...")

    # Keyword test (works without tesseract)
    sample_text = "heritage manuscript ancient calligraphy heritage manuscript text document"
    keywords = extract_keywords(sample_text)
    print(f"  Keywords from sample: {keywords}")
    assert len(keywords) > 0, "Should extract keywords"

    # Script detection test (works without tesseract)
    script = detect_script("Hello World this is English text for testing")
    print(f"  Script detection: {script}")
    assert script["script"] == "Latin", "Should detect Latin script"

    print("  [PASS] OCR Engine Module -- ALL TESTS PASSED")
    return True


def test_aggregator():
    """Test the aggregator module."""
    print("\n" + "=" * 50)
    print("TEST: Aggregator Module")
    print("=" * 50)

    from modules.aggregator import aggregate_results, generate_batch_summary

    prep = [{"filename": "test_001.png"}, {"filename": "test_002.png"}]
    quality = [
        {"quality_score": 85, "blur": {"blur_status": "OK"}, "skew": {"skew_status": "OK"}},
        {"quality_score": 40, "blur": {"blur_status": "BLURRED"}, "skew": {"skew_status": "OK"}},
    ]
    sequence = {"per_file": [
        {"page_number": 1}, {"page_number": 2}
    ], "gap_analysis": {"missing_pages": []}}
    ocr = [{"confidence": 78.5}, {"confidence": 45.0}]

    results = aggregate_results(prep, quality, sequence, ocr)
    summary = generate_batch_summary(results)

    for r in results:
        print(f"  {r['filename']}: score={r['final_score']}, status={r['status']}")

    print(f"  Batch summary: total={summary['total_images']}, "
          f"ready={summary['ready']}, review={summary['review']}, "
          f"rejected={summary['rejected']}, pass_rate={summary['pass_rate']}%")

    assert len(results) == 2, "Should have 2 results"
    assert summary["total_images"] == 2, "Should report 2 images"
    print("  [PASS] Aggregator Module -- ALL TESTS PASSED")
    return True


def test_output_generator():
    """Test the output generator module."""
    print("\n" + "=" * 50)
    print("TEST: Output Generator Module")
    print("=" * 50)

    from modules.output_generator import generate_csv, generate_json_audit
    import tempfile
    import json

    aggregated = [
        {"filename": "test_001.png", "quality_score": 85, "page_number": 1,
         "ocr_confidence": 78.5, "final_score": 85.4, "status": "READY",
         "blur_status": "OK", "skew_status": "OK"},
    ]
    summary = {"total_images": 1, "ready": 1, "review": 0, "rejected": 0}

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "test_output")

    csv_path = os.path.join(output_dir, "test_results.csv")
    json_path = os.path.join(output_dir, "test_audit.json")

    csv_result = generate_csv(aggregated, csv_path)
    json_result = generate_json_audit(aggregated, summary, json_path)

    assert csv_result is not None and os.path.exists(csv_result), "CSV should be generated"
    assert json_result is not None and os.path.exists(json_result), "JSON should be generated"

    # Verify JSON content
    with open(json_result, "r", encoding="utf-8") as f:
        audit = json.load(f)
    assert audit["metadata"]["system"] == "IDP Manuscript Processing System"
    assert len(audit["results"]) == 1

    print(f"  CSV generated: {csv_result}")
    print(f"  JSON generated: {json_result}")
    print(f"  JSON system: {audit['metadata']['system']}")
    print("  [PASS] Output Generator Module -- ALL TESTS PASSED")
    return True


if __name__ == "__main__":
    print("")
    print("=" * 60)
    print("  IDP Manuscript Processing -- Test Suite")
    print("=" * 60)

    passed = 0
    failed = 0
    tests = [
        ("Preprocessor", test_preprocessor),
        ("Quality Checker", test_quality_checker),
        ("Sequence Detector", test_sequence_detector),
        ("OCR Engine", test_ocr_engine),
        ("Aggregator", test_aggregator),
        ("Output Generator", test_output_generator),
    ]

    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    if failed == 0:
        print("  ALL MODULE TESTS PASSED SUCCESSFULLY!")
    else:
        print(f"  WARNING: {failed} test(s) failed")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
