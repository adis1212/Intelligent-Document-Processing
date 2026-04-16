"""
Quality Checker Module — ENHANCED
===================================
Performs deep quality assessment on manuscript images.

Algorithms:
    1. Blur Detection      → Laplacian variance
    2. Skew Detection      → Projection Profile Method (PPM)  ← IMPROVED
    3. Crop Detection      → Border margin content analysis
    4. Occlusion Detection → Dark pixel area ratio
    5. Quality Score       → Weighted composite (0-100)
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger("idp_manuscript.quality_checker")


# ═══════════════════════════════════════════════════════════════
# 1. BLUR DETECTION — Laplacian Variance
# ═══════════════════════════════════════════════════════════════
def check_blur(image, threshold=100.0):
    """
    Detect image blur using Laplacian variance.

    Algorithm:
        - Apply Laplacian operator (edge detector) to the grayscale image.
        - Compute the variance of the resulting map.
        - Sharp images have high variance (many strong edges).
        - Blurry images have low variance (edges are smoothed out).

    Args:
        image   : np.ndarray — grayscale or BGR image
        threshold: float   — variance below this = BLURRED

    Returns:
        dict with blur_score, is_blurred, blur_status
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(laplacian.var())
        is_blurred = variance < threshold

        logger.info(f"Blur check: score={variance:.2f}, threshold={threshold}, blurred={is_blurred}")
        return {
            "blur_score": round(variance, 2),
            "is_blurred": is_blurred,
            "blur_status": "BLURRED" if is_blurred else "OK",
            "threshold_used": threshold,
        }
    except Exception as e:
        logger.error(f"Blur detection failed: {e}")
        return {"blur_score": 0.0, "is_blurred": True, "blur_status": "ERROR", "threshold_used": threshold}


# ═══════════════════════════════════════════════════════════════
# 2. SKEW DETECTION — Projection Profile Method (PPM)
# ═══════════════════════════════════════════════════════════════
def check_skew(image, threshold=3.0, angle_range=(-20, 20), angle_steps=81):
    """
    Detect document skew using the Projection Profile Method (PPM).

    Algorithm (PPM):
        1. Convert image to binary (black text on white).
        2. For each candidate angle in [-20°, +20°]:
           a. Rotate the binary image by that angle.
           b. Compute the horizontal projection profile (row sums).
           c. Compute the variance of the projection profile.
        3. The angle with MAXIMUM variance = best alignment (text in clean rows).
        4. If |best_angle| > threshold → image is SKEWED.

    Why PPM works:
        - When a text document is perfectly horizontal, each row either
          has many dark pixels (text row) or very few (whitespace).
        - This creates a HIGH-variance projection profile.
        - When tilted, text rows bleed across multiple rows → LOW variance.
        - PPM finds the rotation that maximises this row separation.

    Args:
        image       : np.ndarray — grayscale or BGR image
        threshold   : float      — degrees; above this = SKEWED
        angle_range : tuple      — (min_angle, max_angle) search range
        angle_steps : int        — number of angles to test

    Returns:
        dict with skew_angle, is_skewed, skew_status, corrected_image
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        # Binarize — Otsu threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        h, w = binary.shape
        cx, cy = w // 2, h // 2

        # Test angles and score each by projection profile variance
        angles = np.linspace(angle_range[0], angle_range[1], angle_steps)
        scores = []

        for angle in angles:
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            # Horizontal projection profile
            projection = np.sum(rotated, axis=1).astype(np.float64)
            score = float(np.var(projection))
            scores.append(score)

        # Best angle = maximum variance
        best_idx = int(np.argmax(scores))
        best_angle = float(angles[best_idx])
        best_score = scores[best_idx]

        # Skew angle is how far from 0 the best angle is
        skew_magnitude = abs(best_angle)
        is_skewed = skew_magnitude > threshold

        # Generate deskewed image
        M_correct = cv2.getRotationMatrix2D((cx, cy), best_angle, 1.0)
        corrected = cv2.warpAffine(gray, M_correct, (w, h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)

        logger.info(f"Skew (PPM): best_angle={best_angle:.2f}°, variance={best_score:.1f}, skewed={is_skewed}")

        return {
            "skew_angle": round(best_angle, 2),
            "skew_magnitude": round(skew_magnitude, 2),
            "is_skewed": is_skewed,
            "skew_status": "SKEWED" if is_skewed else "OK",
            "corrected_image": corrected,
            "projection_scores": scores,
            "angles_tested": list(angles),
        }

    except Exception as e:
        logger.error(f"Skew detection (PPM) failed: {e}")
        return {
            "skew_angle": 0.0, "skew_magnitude": 0.0,
            "is_skewed": False, "skew_status": "ERROR",
            "corrected_image": None, "projection_scores": [], "angles_tested": [],
        }


# ═══════════════════════════════════════════════════════════════
# 3. CROP DETECTION — Border margin analysis
# ═══════════════════════════════════════════════════════════════
def check_crop(image, margin_percent=2.0):
    """
    Check whether the image has been over-cropped (no clear margins).

    Algorithm:
        1. Binarize image to isolate content (dark pixels).
        2. Sample border strips (top/bottom/left/right = margin_percent%).
        3. If content exists right at the edge → page is likely cropped.

    Returns:
        dict with has_margin, crop_status, margins dict
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        mh = max(1, int(h * margin_percent / 100))
        mw = max(1, int(w * margin_percent / 100))

        top_ok    = np.sum(binary[:mh, :]) == 0
        bottom_ok = np.sum(binary[-mh:, :]) == 0
        left_ok   = np.sum(binary[:, :mw]) == 0
        right_ok  = np.sum(binary[:, -mw:]) == 0

        has_margin = top_ok and bottom_ok and left_ok and right_ok
        return {
            "has_margin": has_margin,
            "crop_status": "OK" if has_margin else "CROPPED",
            "margins": {
                "top_clear": top_ok, "bottom_clear": bottom_ok,
                "left_clear": left_ok, "right_clear": right_ok,
            },
        }
    except Exception as e:
        logger.error(f"Crop detection failed: {e}")
        return {"has_margin": True, "crop_status": "ERROR", "margins": {}}


# ═══════════════════════════════════════════════════════════════
# 4. OCCLUSION DETECTION — Dark pixel ratio
# ═══════════════════════════════════════════════════════════════
def check_occlusion(image, dark_threshold=30, area_percent=15.0):
    """
    Detect occlusion (e.g., thumbs, stamps, water damage) by dark pixel ratio.

    Algorithm:
        1. Check what fraction of pixels are very dark (< dark_threshold).
        2. If that fraction exceeds area_percent% → likely occluded.

    Returns:
        dict with occlusion_percent, is_occluded, occlusion_status
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        dark_px = int(np.sum(gray < dark_threshold))
        total_px = gray.shape[0] * gray.shape[1]
        dark_pct = round((dark_px / total_px) * 100, 2)
        is_occluded = dark_pct > area_percent

        return {
            "occlusion_percent": dark_pct,
            "is_occluded": is_occluded,
            "occlusion_status": "OCCLUDED" if is_occluded else "OK",
        }
    except Exception as e:
        logger.error(f"Occlusion detection failed: {e}")
        return {"occlusion_percent": 0.0, "is_occluded": False, "occlusion_status": "ERROR"}


# ═══════════════════════════════════════════════════════════════
# 5. COMPOSITE QUALITY SCORE
# ═══════════════════════════════════════════════════════════════
def compute_quality_score(blur_result, skew_result, crop_result, occlusion_result):
    """
    Weighted composite quality score (0–100).

    Weights:
        Blur       40 pts  — most critical for OCR
        Skew       25 pts  — affects text alignment
        Crop       20 pts  — affects completeness
        Occlusion  15 pts  — affects readable area
    """
    score = 0.0

    # Blur (40 pts)
    bs = blur_result.get("blur_score", 0)
    score += min(40, 40 * (bs / 500.0))

    # Skew (25 pts)
    sm = skew_result.get("skew_magnitude", 0)
    if sm <= 1.0:
        score += 25
    elif sm <= 5.0:
        score += 25 * (1 - (sm - 1) / 4)
    else:
        score += max(0, 25 * (1 - sm / 20))

    # Crop (20 pts)
    if crop_result.get("has_margin", True):
        score += 20
    else:
        clear = sum(1 for v in crop_result.get("margins", {}).values() if v)
        score += 20 * (clear / 4)

    # Occlusion (15 pts)
    op = occlusion_result.get("occlusion_percent", 0)
    if not occlusion_result.get("is_occluded", False):
        score += 15
    else:
        score += max(0, 15 * (1 - op / 100))

    return round(min(100.0, max(0.0, score)), 1)


# ═══════════════════════════════════════════════════════════════
# 6. RUN FULL QUALITY CHECK
# ═══════════════════════════════════════════════════════════════
def run_quality_check(image, config=None):
    """
    Run all 4 quality checks and compute the composite score.

    Skew impact: If skew is detected (angle > threshold), the overall
    status is capped at "REVIEW" — it can never be "PASS" while skewed.
    This ensures skewed pages are always flagged for manual inspection.

    Returns a dict with full results from every check.
    """
    if config is None:
        config = {}

    skew_threshold = config.get("skew_threshold", 5.0)

    blur_r  = check_blur(image, config.get("blur_threshold", 100.0))
    skew_r  = check_skew(image, skew_threshold)
    crop_r  = check_crop(image, config.get("crop_margin_percent", 2.0))
    occ_r   = check_occlusion(image,
                               config.get("occlusion_dark_threshold", 30),
                               config.get("occlusion_area_percent", 15.0))

    q_score = compute_quality_score(blur_r, skew_r, crop_r, occ_r)
    min_score = config.get("min_quality_score", 50.0)

    # --- Determine overall status with skew override ---
    if q_score >= min_score:
        overall = "PASS"
    else:
        overall = "FAIL"

    # If skew detected, cap status at REVIEW (never PASS while skewed)
    if skew_r.get("is_skewed", False) and overall == "PASS":
        overall = "REVIEW"
        logger.info(f"Skew override: score={q_score} would be PASS, "
                     f"but skew={skew_r['skew_angle']}° > {skew_threshold}° → REVIEW")

    # Store threshold in skew result for dashboard display
    skew_r["threshold_used"] = skew_threshold

    return {
        "blur":           blur_r,
        "skew":           skew_r,
        "crop":           crop_r,
        "occlusion":      occ_r,
        "quality_score":  q_score,
        "overall_status": overall,
    }
