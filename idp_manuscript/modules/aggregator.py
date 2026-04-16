"""
Aggregator Module — Enhanced with Search Index
================================================
Combines results from all pipeline modules into a final assessment.
Generates per-image and batch-level summaries.
Provides metadata-driven document search across processed pages.

Search Features:
    - Inverted index over OCR text, filenames, keywords, quality status
    - Query returns page numbers, filenames, and contextual text snippets
    - Phase 1: English text only. Future: Marathi, Hindi.
"""

import re
import logging
from collections import defaultdict

logger = logging.getLogger("idp_manuscript.aggregator")


def aggregate_results(preprocessing_results, quality_results, sequence_results, ocr_results, config=None):
    """
    Aggregate results from all modules into a final report.

    Score breakdown:
        Quality Score × 0.4  (image quality — blur, skew, crop, occlusion)
        OCR Confidence × 0.4 (text extraction reliability)
        Sequence Bonus × 0.2 (20 pts if page number detected, 0 otherwise)

    Skew impact: If the quality checker flagged the page as REVIEW due to
    skew, the aggregated status is capped at REVIEW (never READY).

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
        quality_status = quality_results[i].get("overall_status", "FAIL") if i < len(quality_results) else "FAIL"

        # Page number
        page_info = sequence_results["per_file"][i] if i < len(sequence_results.get("per_file", [])) else {}
        page_number = page_info.get("page_number", None)

        # OCR confidence
        ocr_conf = ocr_results[i].get("confidence", 0) if i < len(ocr_results) else 0

        # --- Score breakdown ---
        quality_component = round(quality_score * 0.4, 1)
        ocr_component = round(ocr_conf * 0.4, 1)
        sequence_component = 20.0 if page_number is not None else 0.0

        final_score = round(quality_component + ocr_component + sequence_component, 1)

        # Determine status
        if final_score >= ready_threshold:
            status = "READY ✅"
        elif final_score >= review_threshold:
            status = "REVIEW ⚠️"
        else:
            status = "REJECT ❌"

        # --- Skew override: if quality status is REVIEW due to skew,
        #     cap aggregated status at REVIEW (never READY) ---
        skew_info = quality_results[i].get("skew", {}) if i < len(quality_results) else {}
        is_skewed = skew_info.get("is_skewed", False)

        if is_skewed and "READY" in status:
            status = "REVIEW ⚠️"
            logger.info(f"{filename}: Skew override in aggregation → REVIEW")

        # Also cap at REVIEW if quality check said REVIEW or FAIL
        if quality_status in ("REVIEW", "FAIL") and "READY" in status:
            status = "REVIEW ⚠️"

        aggregated.append({
            "filename": filename,
            "quality_score": quality_score,
            "page_number": page_number if page_number is not None else "N/A",
            "ocr_confidence": ocr_conf,
            "final_score": final_score,
            "status": status,
            "blur_status": quality_results[i].get("blur", {}).get("blur_status", "N/A") if i < len(quality_results) else "N/A",
            "skew_status": quality_results[i].get("skew", {}).get("skew_status", "N/A") if i < len(quality_results) else "N/A",
            # Score breakdown for dashboard display
            "quality_component": quality_component,
            "ocr_component": ocr_component,
            "sequence_component": sequence_component,
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


# ═══════════════════════════════════════════════════════════════
# DOCUMENT SEARCH INDEX
# ═══════════════════════════════════════════════════════════════

def build_search_index(ocr_results, preprocessing_results, quality_results, aggregated_results):
    """
    Build a metadata-driven search index across all processed documents.

    Indexes:
        - OCR text (full text per page)
        - Filenames
        - Extracted keywords
        - Page numbers
        - Quality status

    This enables precise keyword-based retrieval across manuscript pages
    and maps results to their exact locations.

    Phase 1: English text only.
    Future: Multilingual search (Marathi, Hindi).

    Args:
        ocr_results: List of OCR result dicts
        preprocessing_results: List of preprocessing result dicts
        quality_results: List of quality result dicts
        aggregated_results: List of aggregated result dicts

    Returns:
        dict: {
            "pages": [...],           # per-page metadata for lookup
            "inverted_index": {...},   # word → [page_indices]
        }
    """
    pages = []
    inverted_index = defaultdict(set)

    for i, ocr_r in enumerate(ocr_results):
        filename = preprocessing_results[i].get("filename", f"image_{i}") if i < len(preprocessing_results) else f"image_{i}"
        page_num = aggregated_results[i].get("page_number", "N/A") if i < len(aggregated_results) else "N/A"
        status = aggregated_results[i].get("status", "N/A") if i < len(aggregated_results) else "N/A"
        quality_status = quality_results[i].get("overall_status", "N/A") if i < len(quality_results) else "N/A"

        text = ocr_r.get("text", "")
        keywords = ocr_r.get("keywords", [])

        page_entry = {
            "index": i,
            "filename": filename,
            "page_number": page_num,
            "text": text,
            "keywords": keywords,
            "status": status,
            "quality_status": quality_status,
        }
        pages.append(page_entry)

        # Index all words from OCR text (lowercase, 3+ chars)
        words = set(re.findall(r'\b[a-zA-Z]{3,}\b', text.lower()))
        for w in words:
            inverted_index[w].add(i)

        # Index keywords specifically
        for kw, _count in keywords:
            inverted_index[kw.lower()].add(i)

        # Index filename tokens
        fname_tokens = re.findall(r'[a-zA-Z]+', filename.lower())
        for t in fname_tokens:
            if len(t) >= 3:
                inverted_index[t].add(i)

    # Convert sets to sorted lists for JSON serialization
    inverted_final = {k: sorted(v) for k, v in inverted_index.items()}

    logger.info(f"Search index built: {len(pages)} pages, {len(inverted_final)} unique terms")
    return {
        "pages": pages,
        "inverted_index": inverted_final,
    }


def search_documents(query, search_index, context_chars=80):
    """
    Search processed documents for a query string.

    Input: user query (e.g., "India")
    Output: matching pages with filenames and contextual text snippets.

    The snippets clearly indicate where the term appears in the document
    by showing surrounding context.

    Args:
        query: Search string
        search_index: Index built by build_search_index()
        context_chars: Number of characters of context around each match

    Returns:
        list of dicts: [{
            "page_index": int,
            "page_number": str,
            "filename": str,
            "status": str,
            "snippets": [str, ...],
            "keyword_match": bool
        }, ...]
    """
    if not query or not search_index:
        return []

    query_lower = query.strip().lower()
    query_words = re.findall(r'\b[a-zA-Z]{2,}\b', query_lower)

    if not query_words:
        return []

    pages = search_index.get("pages", [])
    inverted = search_index.get("inverted_index", {})

    # Find candidate pages from inverted index
    candidate_indices = set()
    for qw in query_words:
        # Exact match
        if qw in inverted:
            candidate_indices.update(inverted[qw])
        # Prefix match for partial queries
        for term, page_list in inverted.items():
            if term.startswith(qw) and len(qw) >= 3:
                candidate_indices.update(page_list)

    # Also do direct text search for multi-word queries
    for i, page in enumerate(pages):
        if query_lower in page["text"].lower():
            candidate_indices.add(i)

    results = []
    for idx in sorted(candidate_indices):
        if idx >= len(pages):
            continue
        page = pages[idx]
        text = page["text"]

        # Extract snippets around query matches
        snippets = _extract_snippets(text, query, context_chars)

        # Check if it's a keyword match
        kw_match = any(query_lower in kw.lower() for kw, _ in page.get("keywords", []))

        if snippets or kw_match:
            results.append({
                "page_index": idx,
                "page_number": page["page_number"],
                "filename": page["filename"],
                "status": page["status"],
                "snippets": snippets if snippets else [f"[Keyword match in: {page['filename']}]"],
                "keyword_match": kw_match,
            })

    logger.info(f"Search '{query}': {len(results)} results found")
    return results


def _extract_snippets(text, query, context_chars=80):
    """
    Extract text snippets surrounding each occurrence of the query.

    Returns:
        list of snippet strings with "..." ellipsis markers
    """
    if not text or not query:
        return []

    snippets = []
    text_lower = text.lower()
    query_lower = query.lower()
    start = 0

    while True:
        pos = text_lower.find(query_lower, start)
        if pos == -1:
            break

        # Context window around the match
        snip_start = max(0, pos - context_chars)
        snip_end = min(len(text), pos + len(query) + context_chars)

        snippet = text[snip_start:snip_end].strip()
        if snip_start > 0:
            snippet = "..." + snippet
        if snip_end < len(text):
            snippet = snippet + "..."

        snippets.append(snippet)
        start = pos + len(query)

        # Limit to 3 snippets per page
        if len(snippets) >= 3:
            break

    return snippets
