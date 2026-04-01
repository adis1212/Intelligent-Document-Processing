"""
Sequence Detector Module — ENHANCED
=====================================
Detects page numbers AND finds duplicate/missing pages.

Algorithms:
    1. Filename-based page number extraction (regex)
    2. Duplicate Detection — perceptual hashing (dHash)
    3. Gap Detection       — missing page numbers
    4. Duplicate Page Detection — content-similarity via image hash
"""

import re
import logging
import numpy as np

logger = logging.getLogger("idp_manuscript.sequence_detector")


# ═══════════════════════════════════════════════════════════════
# 1. PAGE NUMBER EXTRACTION — Regex on filename
# ═══════════════════════════════════════════════════════════════
def extract_page_from_filename(filename, pattern=r"page_(\d+)", fallback_pattern=r"(\d+)"):
    """
    Extract page number from filename using regex.

    Algorithm:
        1. Try the primary pattern (e.g., 'page_001' → 1).
        2. If no match, try the fallback pattern (first number in name).
        3. Return None if nothing found.

    Examples:
        manuscript_page_003.png → 3   (primary pattern)
        scan_0042.tiff          → 42  (fallback pattern)
        untitled.jpg            → None
    """
    try:
        m = re.search(pattern, filename, re.IGNORECASE)
        if m:
            pg = int(m.group(1))
            return {"filename": filename, "page_number": pg, "method": "filename_primary", "status": f"Page {pg}"}

        m = re.search(fallback_pattern, filename)
        if m:
            pg = int(m.group(1))
            return {"filename": filename, "page_number": pg, "method": "filename_fallback", "status": f"Page {pg}"}

        return {"filename": filename, "page_number": None, "method": "not_found", "status": "Unknown"}

    except Exception as e:
        logger.error(f"Page extraction failed for {filename}: {e}")
        return {"filename": filename, "page_number": None, "method": "error", "status": "Error"}


# ═══════════════════════════════════════════════════════════════
# 2. DUPLICATE DETECTION — Difference Hash (dHash)
# ═══════════════════════════════════════════════════════════════
def compute_dhash(image, hash_size=8):
    """
    Compute a perceptual difference hash (dHash) for an image.

    Algorithm (dHash):
        1. Resize the image to (hash_size+1) × hash_size pixels.
        2. Convert to grayscale.
        3. For each row, compare adjacent pixel pairs:
           bit = 1 if left pixel > right pixel, else 0.
        4. Concatenate all bits → a 64-bit fingerprint.

    Why dHash works:
        - Small scaling differences don't affect the hash.
        - The hash captures relative brightness changes, not absolute values.
        - Two visually similar images produce very similar hashes.
        - Hamming distance between hashes measures visual similarity:
          0 = identical, <10 = very similar, >20 = different.

    Args:
        image    : np.ndarray — any image (BGR or grayscale)
        hash_size: int        — determines hash length (hash_size² bits)

    Returns:
        int — the dHash value
    """
    try:
        import cv2
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Resize to (hash_size+1, hash_size) — extra column for comparison
        resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)

        # Compare adjacent pixels across each row
        diff = resized[:, 1:] > resized[:, :-1]      # Boolean mask

        # Convert to integer hash
        hash_val = sum(bool(bit) << i for i, bit in enumerate(diff.flatten()))
        return hash_val

    except Exception as e:
        logger.error(f"dHash computation failed: {e}")
        return None


def hamming_distance(hash1, hash2):
    """
    Compute the Hamming distance between two integer hashes.
    Counts differing bits — lower = more similar.
    """
    if hash1 is None or hash2 is None:
        return 64   # Max distance for 64-bit hash
    xor = hash1 ^ hash2
    return bin(xor).count("1")


def detect_duplicates(images_data, similarity_threshold=10):
    """
    Detect duplicate or near-duplicate images using dHash + Hamming distance.

    Algorithm:
        1. Compute dHash for every image.
        2. Compare every pair (i, j) where i < j.
        3. If Hamming(hash_i, hash_j) ≤ threshold → mark as duplicates.

    Args:
        images_data        : list of dicts {"filename": str, "image": np.ndarray}
        similarity_threshold: int — max Hamming distance to consider duplicate

    Returns:
        dict with:
            - hashes     : per-image hash values
            - duplicates : list of duplicate pairs [{pair, distance, filenames}]
            - dup_flags  : per-filename duplicate flag
    """
    hashes = {}
    for item in images_data:
        h = compute_dhash(item["image"])
        hashes[item["filename"]] = h

    duplicates = []
    dup_filenames = set()
    filenames = [item["filename"] for item in images_data]

    for i in range(len(filenames)):
        for j in range(i + 1, len(filenames)):
            fn_i, fn_j = filenames[i], filenames[j]
            dist = hamming_distance(hashes.get(fn_i), hashes.get(fn_j))
            if dist <= similarity_threshold:
                duplicates.append({
                    "pair": (fn_i, fn_j),
                    "hamming_distance": dist,
                    "similarity_percent": round((1 - dist / 64) * 100, 1),
                })
                dup_filenames.update([fn_i, fn_j])
                logger.warning(f"Duplicate detected: {fn_i} ↔ {fn_j} (dist={dist})")

    dup_flags = {fn: (fn in dup_filenames) for fn in filenames}

    logger.info(f"Duplicate detection: {len(duplicates)} duplicate pairs found")
    return {
        "hashes": {fn: str(h) for fn, h in hashes.items()},
        "duplicates": duplicates,
        "dup_flags": dup_flags,
        "total_duplicates": len(duplicates),
    }


# ═══════════════════════════════════════════════════════════════
# 3. GAP DETECTION — Missing & out-of-order pages
# ═══════════════════════════════════════════════════════════════
def detect_gaps(page_numbers):
    """
    Detect missing pages and duplicate page numbers in the sequence.

    Algorithm:
        1. Filter out None values, sort remaining page numbers.
        2. Build expected set = range(min, max+1).
        3. Missing = expected - actual.
        4. Duplicates = numbers appearing more than once.

    Args:
        page_numbers: list of int or None values

    Returns:
        dict with missing_pages, duplicates, range, total_detected
    """
    valid = [p for p in page_numbers if p is not None]
    if len(valid) < 2:
        return {"missing_pages": [], "duplicates": [], "range": (0, 0),
                "total_detected": len(valid), "is_complete": True}

    sorted_pages = sorted(valid)
    min_pg, max_pg = sorted_pages[0], sorted_pages[-1]

    expected = set(range(min_pg, max_pg + 1))
    actual   = set(sorted_pages)

    missing    = sorted(expected - actual)
    seen       = {}
    dup_nums   = []
    for p in sorted_pages:
        seen[p] = seen.get(p, 0) + 1
    dup_nums = sorted([p for p, cnt in seen.items() if cnt > 1])

    logger.info(f"Gap analysis: range={min_pg}-{max_pg}, missing={len(missing)}, dups={len(dup_nums)}")

    return {
        "missing_pages": missing,
        "duplicates": dup_nums,
        "range": (min_pg, max_pg),
        "total_detected": len(valid),
        "is_complete": len(missing) == 0 and len(dup_nums) == 0,
        "expected_count": max_pg - min_pg + 1,
    }


# ═══════════════════════════════════════════════════════════════
# 4. MAIN RUNNER
# ═══════════════════════════════════════════════════════════════
def run_sequence_detection(filenames, config=None, images_data=None):
    """
    Run complete sequence detection pipeline.

    Args:
        filenames   : list of image filenames
        config      : sequence detection config dict
        images_data : optional list of {"filename":str, "image":np.ndarray}
                      for duplicate content detection

    Returns:
        dict with per_file results, gap_analysis, duplicate_analysis
    """
    if config is None:
        config = {}

    pattern  = config.get("filename_pattern", r"page_(\d+)")
    fallback = config.get("fallback_pattern", r"(\d+)")

    # Per-file page detection
    per_file    = []
    page_nums   = []
    for fn in filenames:
        result = extract_page_from_filename(fn, pattern, fallback)
        per_file.append(result)
        page_nums.append(result["page_number"])

    # Gap analysis
    gap_analysis = detect_gaps(page_nums)

    # Duplicate image content detection (if images provided)
    dup_analysis = None
    if images_data:
        dup_analysis = detect_duplicates(images_data, similarity_threshold=10)
        # Annotate per_file with duplicate flags
        dup_flags = dup_analysis.get("dup_flags", {})
        for item in per_file:
            item["is_duplicate"] = dup_flags.get(item["filename"], False)
    else:
        for item in per_file:
            item["is_duplicate"] = False

    return {
        "per_file":         per_file,
        "gap_analysis":     gap_analysis,
        "duplicate_analysis": dup_analysis,
    }
