"""
IDP Manuscript Processing System - Main Pipeline
==================================================
Orchestrates the complete document processing pipeline.
"""

import os
import sys
import logging
from utils.config_loader import load_config
from utils.logger import setup_logger
from modules.preprocessor import load_batch, preprocess_image
from modules.quality_checker import run_quality_check
from modules.sequence_detector import run_sequence_detection
from modules.ocr_engine import run_ocr
from modules.aggregator import aggregate_results, generate_batch_summary
from modules.output_generator import generate_csv, generate_json_audit

logger = setup_logger()


def run_pipeline(folder_path, progress_callback=None):
    """
    Run the complete IDP pipeline.

    Args:
        folder_path: Path to folder containing manuscript images
        progress_callback: Optional callback function for progress updates
                          Signature: callback(step_name, status, data)

    Returns:
        dict with all pipeline results
    """
    config = load_config()
    results = {
        "preprocessing": [],
        "quality": [],
        "sequence": None,
        "ocr": [],
        "aggregated": [],
        "batch_summary": {},
    }

    def update_progress(step, status, data=None):
        if progress_callback:
            progress_callback(step, status, data)

    try:
        # ── Step 1: Load & Preprocess ───────────────────────────────
        update_progress("preprocessing", "running")
        logger.info(f"Starting pipeline for: {folder_path}")

        images = load_batch(folder_path, config["preprocessing"]["supported_formats"])
        if not images:
            update_progress("preprocessing", "error", {"error": "No images found"})
            logger.error("No images found in the specified folder")
            return results

        preprocess_config = config["preprocessing"]
        for img_data in images:
            processed = preprocess_image(img_data["image"], preprocess_config)
            processed["filename"] = img_data["filename"]
            processed["path"] = img_data["path"]
            results["preprocessing"].append(processed)

        update_progress("preprocessing", "completed",
                       {"total_images": len(results["preprocessing"])})

        # ── Step 2: Quality Check ───────────────────────────────────
        update_progress("quality_check", "running")

        quality_config = config["quality"]
        for prep_result in results["preprocessing"]:
            img = prep_result.get("processed", prep_result.get("original"))
            quality = run_quality_check(img, quality_config)
            results["quality"].append(quality)

        update_progress("quality_check", "completed")

        # ── Step 3: Sequence Detection ──────────────────────────────
        update_progress("sequence_detection", "running")

        filenames = [p["filename"] for p in results["preprocessing"]]
        seq_config = config["sequence"]
        results["sequence"] = run_sequence_detection(filenames, seq_config)

        update_progress("sequence_detection", "completed",
                       results["sequence"]["gap_analysis"])

        # ── Step 4: OCR Processing ──────────────────────────────────
        update_progress("ocr_processing", "running")

        ocr_config = config["ocr"]
        for prep_result in results["preprocessing"]:
            img = prep_result.get("processed", prep_result.get("original"))
            ocr_result = run_ocr(img, ocr_config)
            results["ocr"].append(ocr_result)

        update_progress("ocr_processing", "completed")

        # ── Step 5: Aggregation ─────────────────────────────────────
        update_progress("aggregation", "running")

        agg_config = config["aggregation"]
        results["aggregated"] = aggregate_results(
            results["preprocessing"],
            results["quality"],
            results["sequence"],
            results["ocr"],
            agg_config,
        )
        results["batch_summary"] = generate_batch_summary(results["aggregated"])

        update_progress("aggregation", "completed", results["batch_summary"])

        # ── Step 6: Output Generation ───────────────────────────────
        update_progress("output_generation", "running")

        output_dir = config["output"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(output_dir, config["output"]["csv_filename"])
        json_path = os.path.join(output_dir, config["output"]["json_filename"])

        results["csv_path"] = generate_csv(results["aggregated"], csv_path)
        results["json_path"] = generate_json_audit(
            results["aggregated"], results["batch_summary"], json_path
        )

        update_progress("output_generation", "completed")

        logger.info("Pipeline completed successfully!")
        return results

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    # CLI usage
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_folder_path>")
        sys.exit(1)

    folder = sys.argv[1]
    results = run_pipeline(folder)

    print(f"\n{'='*50}")
    print("PIPELINE RESULTS")
    print(f"{'='*50}")
    summary = results.get("batch_summary", {})
    print(f"Total Images:   {summary.get('total_images', 0)}")
    print(f"Ready:          {summary.get('ready', 0)}")
    print(f"Review:         {summary.get('review', 0)}")
    print(f"Rejected:       {summary.get('rejected', 0)}")
    print(f"Pass Rate:      {summary.get('pass_rate', 0)}%")
