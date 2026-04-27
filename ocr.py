"""
OCR engine wrapper — performance-optimized.

Key optimizations:
  • Reader is initialized ONCE per process (singleton + Streamlit cache)
  • paragraph=False keeps detection granular for line-level confidence
  • detail=1 gives us bbox + confidence in one pass
"""
from __future__ import annotations

import logging
from typing import List, Dict
import numpy as np

log = logging.getLogger(__name__)

_reader = None


def _get_reader():
    """Lazy singleton. Streamlit also wraps this with @st.cache_resource."""
    global _reader
    if _reader is None:
        try:
            import easyocr
            log.info("Initializing EasyOCR reader (English)...")
            _reader = easyocr.Reader(['en'], gpu=False, verbose=False,
                                     download_enabled=True)
        except ImportError as exc:
            raise ImportError(
                "EasyOCR is not installed. Run: pip install easyocr"
            ) from exc
    return _reader


def run_ocr(image: np.ndarray) -> List[Dict]:
    """Returns list of {text, confidence, bbox, y_center}, sorted top→bottom."""
    reader = _get_reader()
    try:
        # paragraph=False → line-level granularity
        # detail=1        → returns (bbox, text, confidence) tuples
        raw = reader.readtext(image, detail=1, paragraph=False,
                              batch_size=8, workers=0)
    except Exception as exc:
        log.error("OCR failed: %s", exc)
        return []

    lines = []
    for entry in raw:
        if len(entry) != 3:
            continue
        bbox, text, conf = entry
        if not text or not text.strip():
            continue
        y_center = float(np.mean([pt[1] for pt in bbox]))
        lines.append({
            "text": text.strip(),
            "confidence": float(conf),
            "bbox": [[float(p[0]), float(p[1])] for p in bbox],
            "y_center": y_center,
        })
    lines.sort(key=lambda l: l["y_center"])
    return lines


def get_full_text(lines: List[Dict]) -> str:
    return "\n".join(line["text"] for line in lines)


def get_average_confidence(lines: List[Dict]) -> float:
    if not lines:
        return 0.0
    return float(np.mean([l["confidence"] for l in lines]))
