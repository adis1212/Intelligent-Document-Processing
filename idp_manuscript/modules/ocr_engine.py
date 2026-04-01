"""
OCR Module
===========
Performs Optical Character Recognition on manuscript images.

Features:
    - Text extraction using pytesseract
    - Confidence scoring
    - Script/language detection
    - Keyword extraction
"""

import logging
import re
from collections import Counter

logger = logging.getLogger("idp_manuscript.ocr_engine")

# Try to import pytesseract - gracefully handle if not available
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not available. OCR will use fallback mode.")


def extract_text(image, config=None):
    """
    Extract text from an image using OCR.

    Args:
        image: Preprocessed grayscale image (np.ndarray)
        config: OCR configuration dict

    Returns:
        dict: {"text": str, "confidence": float, "word_count": int}
    """
    if config is None:
        config = {}

    language = config.get("language", "eng")
    psm_mode = config.get("psm_mode", 6)
    oem_mode = config.get("oem_mode", 3)

    try:
        if not TESSERACT_AVAILABLE:
            return _fallback_ocr(image)

        custom_config = f"--oem {oem_mode} --psm {psm_mode}"

        # Get full data with confidence scores
        data = pytesseract.image_to_data(
            image, lang=language, config=custom_config,
            output_type=pytesseract.Output.DICT
        )

        # Extract text
        text_parts = []
        confidences = []

        for i, word in enumerate(data["text"]):
            conf = int(data["conf"][i])
            if conf > 0 and word.strip():
                text_parts.append(word)
                confidences.append(conf)

        full_text = " ".join(text_parts)
        avg_confidence = round(sum(confidences) / len(confidences), 1) if confidences else 0.0
        word_count = len(text_parts)

        logger.info(f"OCR: extracted {word_count} words, confidence={avg_confidence:.1f}%")

        return {
            "text": full_text,
            "confidence": avg_confidence,
            "word_count": word_count,
            "success": True,
        }

    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return _fallback_ocr(image, error=str(e))


def _fallback_ocr(image, error=None):
    """Fallback when Tesseract is not available."""
    msg = "Tesseract not available" if error is None else f"OCR Error: {error}"
    logger.warning(f"Using fallback OCR mode: {msg}")
    return {
        "text": f"[OCR Fallback] {msg}. Install Tesseract for actual text extraction.",
        "confidence": 0.0,
        "word_count": 0,
        "success": False,
    }


def extract_keywords(text, top_n=10):
    """
    Extract top keywords from OCR text.

    Args:
        text: Extracted text string
        top_n: Number of top keywords to return

    Returns:
        list of (word, count) tuples
    """
    if not text or len(text) < 5:
        return []

    # Clean text
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

    # Remove common stop words
    stop_words = {
        "the", "and", "was", "for", "are", "but", "not", "you",
        "all", "can", "had", "her", "one", "our", "out", "has",
        "his", "how", "its", "may", "new", "now", "old", "see",
        "way", "who", "did", "get", "let", "say", "she", "too",
        "use", "from", "with", "this", "that", "they", "been",
        "have", "many", "some", "them", "than", "each", "make",
        "like", "long", "look", "come", "could", "more", "these",
        "into", "other", "which", "their", "there", "about",
    }

    filtered = [w for w in words if w not in stop_words]
    counter = Counter(filtered)

    return counter.most_common(top_n)


def detect_script(text):
    """
    Detect the script/writing system of the text.

    Returns:
        dict: {"script": str, "confidence": str}
    """
    if not text or len(text) < 5:
        return {"script": "Unknown", "confidence": "Low"}

    # Simple heuristic-based detection
    latin_count = len(re.findall(r'[a-zA-Z]', text))
    devanagari_count = len(re.findall(r'[\u0900-\u097F]', text))
    arabic_count = len(re.findall(r'[\u0600-\u06FF]', text))
    cjk_count = len(re.findall(r'[\u4E00-\u9FFF]', text))
    total = len(text)

    if total == 0:
        return {"script": "Unknown", "confidence": "Low"}

    scores = {
        "Latin": latin_count / total,
        "Devanagari": devanagari_count / total,
        "Arabic": arabic_count / total,
        "CJK": cjk_count / total,
    }

    best_script = max(scores, key=scores.get)
    best_ratio = scores[best_script]

    if best_ratio > 0.5:
        confidence = "High"
    elif best_ratio > 0.2:
        confidence = "Medium"
    else:
        confidence = "Low"
        best_script = "Mixed/Unknown"

    return {"script": best_script, "confidence": confidence}


def run_ocr(image, config=None):
    """
    Run complete OCR pipeline on an image.

    Args:
        image: Preprocessed image
        config: OCR configuration

    Returns:
        dict with text, confidence, keywords, and script info
    """
    if config is None:
        config = {}

    # Extract text
    extraction = extract_text(image, config)

    # Extract keywords
    keywords = extract_keywords(extraction["text"])

    # Detect script
    script = detect_script(extraction["text"])

    return {
        "text": extraction["text"],
        "confidence": extraction["confidence"],
        "word_count": extraction["word_count"],
        "keywords": keywords,
        "script": script,
        "success": extraction["success"],
    }
