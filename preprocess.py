"""
Image preprocessing module.

Prepares receipt images for OCR by handling:
  - Noise (fast non-local means denoising)
  - Blur detection (Laplacian variance)
  - Skew / rotation (Hough lines + minAreaRect)
  - Lighting / contrast (CLAHE on the L-channel of LAB)

Each step is intentionally conservative — over-processing destroys text edges.
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple


# ── Quality metrics ──────────────────────────────────────────────────────────
def estimate_blur(gray: np.ndarray) -> float:
    """
    Laplacian-variance blur score.
    Higher = sharper. <100 typically means the image is blurry.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_brightness(gray: np.ndarray) -> float:
    """Mean pixel intensity (0-255). <50 = very dark, >200 = washed out."""
    return float(np.mean(gray))


# ── Individual preprocessing steps ───────────────────────────────────────────
def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    CLAHE on the L-channel of LAB color space.
    Fixes uneven lighting (e.g. shadow across the receipt) without
    over-saturating colors.
    """
    if len(image.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def denoise(image: np.ndarray) -> np.ndarray:
    """Edge-preserving denoising — keeps text crisp."""
    if len(image.shape) == 2:
        return cv2.fastNlMeansDenoising(image, None, h=10,
                                        templateWindowSize=7,
                                        searchWindowSize=21)
    return cv2.fastNlMeansDenoisingColored(image, None,
                                           h=10, hColor=10,
                                           templateWindowSize=7,
                                           searchWindowSize=21)


def deskew(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Detect skew angle using minAreaRect on a binarized version,
    then rotate to straighten. Returns (rotated_image, angle_degrees).
    """
    gray = to_grayscale(image)
    # Binarize (inverted so text becomes white blobs)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 50:
        return image, 0.0

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Skip rotation for sub-degree skew (avoid resampling artifacts)
    if abs(angle) < 0.5:
        return image, 0.0

    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, float(angle)


def resize_for_ocr(image: np.ndarray,
                   target_height: int = 1200,
                   max_height:    int = 1800) -> np.ndarray:
    """
    Resize image for OCR.
      • Upscale small images (< target_height) — more pixels per char helps OCR
      • Downscale huge images (> max_height) — speeds up OCR by 3-5×

    1200px is the sweet spot: OCR accuracy plateaus, runtime drops.
    """
    h, w = image.shape[:2]
    if h < target_height:
        scale = target_height / h
        return cv2.resize(image, (int(w * scale), target_height),
                          interpolation=cv2.INTER_CUBIC)
    if h > max_height:
        scale = max_height / h
        return cv2.resize(image, (int(w * scale), max_height),
                          interpolation=cv2.INTER_AREA)
    return image


# ── Full pipeline ────────────────────────────────────────────────────────────
def preprocess(image_path: str | Path,
               return_metrics: bool = False) -> np.ndarray | tuple:
    """
    End-to-end preprocessing pipeline.

    Args:
        image_path: path to the receipt image.
        return_metrics: if True, also return a metrics dict.

    Returns:
        Preprocessed BGR image (np.ndarray), or (image, metrics) if
        return_metrics=True.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    metrics = {
        "original_shape": img.shape,
        "blur_score": estimate_blur(to_grayscale(img)),
        "brightness": estimate_brightness(to_grayscale(img)),
    }

    # 1. Upscale tiny images (helps OCR a lot)
    img = resize_for_ocr(img, target_height=1500)

    # 2. Fix uneven lighting
    img = enhance_contrast(img)

    # 3. Denoise (mild — preserves text)
    img = denoise(img)

    # 4. Straighten if skewed
    img, skew_angle = deskew(img)
    metrics["skew_angle"] = skew_angle
    metrics["processed_shape"] = img.shape

    if return_metrics:
        return img, metrics
    return img


def save_preview(original_path: str | Path,
                 processed: np.ndarray,
                 out_path: str | Path) -> None:
    """Save a side-by-side before/after image for documentation."""
    original = cv2.imread(str(original_path))
    if original is None:
        return
    # Resize both to same height for clean side-by-side
    h = 600
    o_w = int(original.shape[1] * h / original.shape[0])
    p_w = int(processed.shape[1] * h / processed.shape[0])
    o_resized = cv2.resize(original, (o_w, h))
    p_resized = cv2.resize(processed, (p_w, h))
    combined = np.hstack([o_resized, p_resized])
    cv2.imwrite(str(out_path), combined)
