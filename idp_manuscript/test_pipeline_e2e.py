"""End-to-end pipeline test — verifies all 6 modules work together."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config_loader import load_config
from utils.sample_generator import generate_sample_manuscripts
from modules.preprocessor import load_batch, preprocess_image
from modules.quality_checker import run_quality_check
from modules.sequence_detector import run_sequence_detection
from modules.ocr_engine import run_ocr
from modules.aggregator import (
    aggregate_results, generate_batch_summary,
    build_search_index, search_documents,
)
from modules.output_generator import generate_csv, generate_json_audit


def main():
    passed = True

    # Step 0: Generate samples
    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "data", "sample_images")
    files = generate_sample_manuscripts(sample_dir, 6)
    print(f"[OK] Generated {len(files)} sample images")

    # Step 1: Load & preprocess
    config = load_config()
    imgs = load_batch(sample_dir, config["preprocessing"]["supported_formats"])
    print(f"[OK] Loaded {len(imgs)} images")

    prep_results = []
    for img_data in imgs:
        p = preprocess_image(img_data["image"], config["preprocessing"])
        p["filename"] = img_data["filename"]
        p["path"] = img_data["path"]
        prep_results.append(p)
    print(f"[OK] Preprocessed {len(prep_results)} images")

    # Step 2: Quality check
    qual_results = []
    for prep in prep_results:
        img = prep.get("processed", prep.get("original"))
        qual_results.append(run_quality_check(img, config["quality"]))
    print(f"[OK] Quality checked {len(qual_results)} images")

    for i, qr in enumerate(qual_results):
        fn = prep_results[i]["filename"]
        angle = qr["skew"]["skew_angle"]
        skewed = qr["skew"]["is_skewed"]
        thresh = qr["skew"].get("threshold_used", "?")
        status = qr["overall_status"]
        print(f"     {fn}: score={qr['quality_score']}, "
              f"status={status}, skew={angle}deg (thresh={thresh}), "
              f"is_skewed={skewed}")

        # VERIFY: if skewed, status should NOT be PASS
        if skewed and status == "PASS":
            print(f"  [FAIL] Skew override not working for {fn}!")
            passed = False

    # Step 3: Sequence detection
    fnames = [p["filename"] for p in prep_results]
    images_data = [{"filename": p["filename"],
                    "image": p.get("processed", p.get("original"))}
                   for p in prep_results]
    seq_results = run_sequence_detection(fnames, config["sequence"],
                                         images_data=images_data)
    gap = seq_results["gap_analysis"]
    print(f"[OK] Sequence: {gap['total_detected']} pages, "
          f"missing={gap['missing_pages']}, complete={gap['is_complete']}")

    # Step 4: OCR
    ocr_results = []
    for prep in prep_results:
        img = prep.get("processed", prep.get("original"))
        ocr_results.append(run_ocr(img, config["ocr"]))
    print(f"[OK] OCR processed {len(ocr_results)} images")

    for i, ocr_r in enumerate(ocr_results):
        fn = prep_results[i]["filename"]
        print(f"     {fn}: conf={ocr_r['confidence']}%, "
              f"words={ocr_r['word_count']}, "
              f"script={ocr_r['script']['script']}, "
              f"success={ocr_r['success']}")

        # VERIFY: confidence should be > 0
        if ocr_r["confidence"] <= 0:
            print(f"  [FAIL] OCR confidence is 0 for {fn}!")
            passed = False

        # VERIFY: should not contain "Tesseract not available" error msg
        if "Tesseract not available" in ocr_r["text"]:
            print(f"  [FAIL] Still showing Tesseract error for {fn}!")
            passed = False

        # VERIFY: should have keywords
        if not ocr_r["keywords"]:
            print(f"  [WARN] No keywords for {fn}")

    # Step 5: Aggregation
    agg = aggregate_results(prep_results, qual_results, seq_results,
                            ocr_results, config["aggregation"])
    batch = generate_batch_summary(agg)
    print(f"[OK] Aggregation: total={batch['total_images']}, "
          f"ready={batch['ready']}, review={batch['review']}, "
          f"rejected={batch['rejected']}, pass_rate={batch['pass_rate']}%")

    # VERIFY: score breakdown fields exist
    for a in agg:
        if "quality_component" not in a:
            print(f"  [FAIL] Missing quality_component in {a['filename']}!")
            passed = False
        if "ocr_component" not in a:
            print(f"  [FAIL] Missing ocr_component in {a['filename']}!")
            passed = False

    # Step 5b: Search
    idx = build_search_index(ocr_results, prep_results, qual_results, agg)
    print(f"[OK] Search index: {len(idx['pages'])} pages, "
          f"{len(idx['inverted_index'])} unique terms")

    # Test search
    for query in ["India", "manuscript", "calligraphy", "heritage"]:
        results = search_documents(query, idx)
        print(f"     Search '{query}': {len(results)} results")
        for r in results[:2]:
            snippet = r["snippets"][0][:80] if r["snippets"] else "no snippet"
            print(f"       Page {r['page_number']} ({r['filename']}): {snippet}...")

    # Step 6: Output
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = generate_csv(agg, os.path.join(out_dir, "processing_results.csv"))
    json_path = generate_json_audit(agg, batch,
                                     os.path.join(out_dir, "audit_log.json"))
    print(f"[OK] CSV: {csv_path}")
    print(f"[OK] JSON: {json_path}")

    if csv_path and os.path.exists(csv_path):
        size = os.path.getsize(csv_path)
        print(f"     CSV size: {size} bytes")
    if json_path and os.path.exists(json_path):
        size = os.path.getsize(json_path)
        print(f"     JSON size: {size} bytes")

    print()
    if passed:
        print("=" * 50)
        print("ALL PIPELINE STEPS PASSED")
        print("=" * 50)
    else:
        print("=" * 50)
        print("SOME CHECKS FAILED — see above")
        print("=" * 50)
        sys.exit(1)


if __name__ == "__main__":
    main()
