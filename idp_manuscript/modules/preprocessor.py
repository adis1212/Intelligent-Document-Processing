"""
Preprocessor Module
====================
Handles image loading and preprocessing for the IDP pipeline.

Functions:
    - load_batch(): Load all images from a directory
    - preprocess_image(): Apply preprocessing (resize, denoise, contrast)
"""

import os
import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger("idp_manuscript.preprocessor")


def load_batch(folder_path, supported_formats=None):
    """
    Load all supported images from a folder.

    Args:
        folder_path: Path to the image folder
        supported_formats: List of supported file extensions

    Returns:
        List of dicts: [{"filename": str, "path": str, "image": np.ndarray}, ...]
    """
    if supported_formats is None:
        supported_formats = [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"]

    if not os.path.isdir(folder_path):
        logger.error(f"Directory not found: {folder_path}")
        raise FileNotFoundError(f"Directory not found: {folder_path}")

    images = []
    files = sorted(os.listdir(folder_path))

    for filename in files:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported_formats:
            continue

        filepath = os.path.join(folder_path, filename)
        try:
            img = cv2.imread(filepath)
            if img is None:
                logger.warning(f"Could not read image: {filename}")
                continue

            images.append({
                "filename": filename,
                "path": filepath,
                "image": img,
            })
            logger.info(f"Loaded: {filename} ({img.shape[1]}x{img.shape[0]})")

        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            continue

    logger.info(f"Loaded {len(images)} images from {folder_path}")
    return images


def preprocess_image(image, config=None):
    """
    Apply preprocessing pipeline to a single image.

    Steps:
        1. Convert to grayscale
        2. Resize to target dimensions
        3. Denoise using Non-Local Means
        4. Enhance contrast using CLAHE

    Args:
        image: np.ndarray (BGR)
        config: dict with preprocessing parameters

    Returns:
        dict with original and processed images + metadata
    """
    if config is None:
        config = {
            "target_size": [1024, 1024],
            "denoise_strength": 10,
            "contrast_clip_limit": 2.0,
            "contrast_tile_size": [8, 8],
        }

    result = {
        "original": image.copy(),
        "steps": {},
    }

    try:
        # Step 1: Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        result["steps"]["grayscale"] = gray.copy()

        # Step 2: Resize
        target_w, target_h = config.get("target_size", [1024, 1024])
        h, w = gray.shape[:2]
        scale = min(target_w / w, target_h / h)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = gray.copy()
        result["steps"]["resized"] = resized.copy()

        # Step 3: Denoise
        denoise_strength = config.get("denoise_strength", 10)
        denoised = cv2.fastNlMeansDenoising(
            resized, None, denoise_strength, 7, 21
        )
        result["steps"]["denoised"] = denoised.copy()

        # Step 4: Contrast Enhancement (CLAHE)
        clip_limit = config.get("contrast_clip_limit", 2.0)
        tile_size = tuple(config.get("contrast_tile_size", [8, 8]))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        enhanced = clahe.apply(denoised)
        result["steps"]["enhanced"] = enhanced.copy()

        # Final result
        result["processed"] = enhanced
        result["original_size"] = (w, h)
        result["processed_size"] = enhanced.shape[:2][::-1]
        result["success"] = True

        logger.info(f"Preprocessing complete: {w}x{h} → {enhanced.shape[1]}x{enhanced.shape[0]}")

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        result["processed"] = image
        result["success"] = False
        result["error"] = str(e)

    return result
