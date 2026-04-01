"""
Output Generator Module
========================
Generates CSV reports and JSON audit logs from pipeline results.
"""

import os
import csv
import json
import logging
from datetime import datetime

logger = logging.getLogger("idp_manuscript.output_generator")


def generate_csv(aggregated_results, output_path):
    """
    Generate CSV report from aggregated results.

    Args:
        aggregated_results: List of result dicts
        output_path: Path to save CSV file

    Returns:
        str: Path to generated CSV file
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if not aggregated_results:
            logger.warning("No results to write to CSV")
            return None

        fieldnames = [
            "filename", "quality_score", "page_number",
            "ocr_confidence", "final_score", "status",
            "blur_status", "skew_status"
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(aggregated_results)

        logger.info(f"CSV report generated: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"CSV generation failed: {e}")
        return None


def generate_json_audit(aggregated_results, batch_summary, output_path):
    """
    Generate JSON audit log with full pipeline metadata.

    Args:
        aggregated_results: List of result dicts
        batch_summary: Batch summary statistics
        output_path: Path to save JSON file

    Returns:
        str: Path to generated JSON file
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        audit_log = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "system": "IDP Manuscript Processing System",
                "version": "1.0.0",
            },
            "batch_summary": batch_summary,
            "results": aggregated_results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(audit_log, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"JSON audit log generated: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"JSON audit generation failed: {e}")
        return None
