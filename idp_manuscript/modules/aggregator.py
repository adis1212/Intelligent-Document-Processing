"""
Aggregator Module
==================
Combines results from all pipeline modules into a final assessment.
Generates per-image and batch-level summaries.
"""

import logging

logger = logging.getLogger("idp_manuscript.aggregator")


def aggregate_results(preprocessing_results, quality_results, sequence_results, ocr_results, config=None):
    """
    Aggregate results from all modules into a final report.

    Args:
        preprocessing_results: List of preprocessing dicts
        quality_results: List of quality check dicts
        sequence_results: Sequence detection results
        ocr_results: List of OCR result dicts
        config: Aggregation configuration

    Returns:
        list of dicts with combined per-image results
    """
    if config is None:
        config = {}

    ready_threshold = config.get("ready_threshold", 70.0)
    review_threshold = config.get("review_threshold", 40.0)

    aggregated = []

    for i in range(len(preprocessing_results)):
        filename = preprocessing_results[i].get("filename", f"image_{i}")

        # Quality score
        quality_score = quality_results[i].get("quality_score", 0) if i < len(quality_results) else 0

        # Page number
        page_info = sequence_results["per_file"][i] if i < len(sequence_results.get("per_file", [])) else {}
        page_number = page_info.get("page_number", None)

        # OCR confidence
        ocr_conf = ocr_results[i].get("confidence", 0) if i < len(ocr_results) else 0

        # Calculate final score (weighted average)
        final_score = round(
            (quality_score * 0.4) + (ocr_conf * 0.4) + (20 if page_number is not None else 0),
            1
        )

        # Determine status
        if final_score >= ready_threshold:
            status = "READY ✅"
        elif final_score >= review_threshold:
            status = "REVIEW ⚠️"
        else:
            status = "REJECT ❌"

        aggregated.append({
            "filename": filename,
            "quality_score": quality_score,
            "page_number": page_number if page_number is not None else "N/A",
            "ocr_confidence": ocr_conf,
            "final_score": final_score,
            "status": status,
            "blur_status": quality_results[i].get("blur", {}).get("blur_status", "N/A") if i < len(quality_results) else "N/A",
            "skew_status": quality_results[i].get("skew", {}).get("skew_status", "N/A") if i < len(quality_results) else "N/A",
        })

    logger.info(f"Aggregation complete: {len(aggregated)} images processed")
    return aggregated


def generate_batch_summary(aggregated_results):
    """
    Generate summary statistics for the entire batch.

    Returns:
        dict with batch-level statistics
    """
    total = len(aggregated_results)
    if total == 0:
        return {"total": 0}

    ready_count = sum(1 for r in aggregated_results if "READY" in r.get("status", ""))
    review_count = sum(1 for r in aggregated_results if "REVIEW" in r.get("status", ""))
    reject_count = sum(1 for r in aggregated_results if "REJECT" in r.get("status", ""))

    avg_quality = round(sum(r.get("quality_score", 0) for r in aggregated_results) / total, 1)
    avg_ocr = round(sum(r.get("ocr_confidence", 0) for r in aggregated_results) / total, 1)
    avg_final = round(sum(r.get("final_score", 0) for r in aggregated_results) / total, 1)

    return {
        "total_images": total,
        "ready": ready_count,
        "review": review_count,
        "rejected": reject_count,
        "avg_quality_score": avg_quality,
        "avg_ocr_confidence": avg_ocr,
        "avg_final_score": avg_final,
        "pass_rate": round((ready_count / total) * 100, 1),
    }
