"""
Confidence scoring & reliability layer.

Combines three signals into a single 0-1 confidence score per field:

  1. OCR confidence — how sure the OCR engine was about reading the text.
  2. Pattern validation — does the value match the expected format?
     (e.g. dates parse cleanly, currency values are in plausible ranges)
  3. Heuristic context — was the value found via a strong keyword anchor
     ("Grand Total: ...") or a weaker fallback (largest number on receipt)?

Final confidence = ocr_conf × validation_factor × heuristic_factor

The heuristic_factor is already baked into the raw confidence returned by
the extractor functions, so this module multiplies in the validation_factor
on top of that.

Low-confidence fields (<0.7) are flagged for human review.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional


VALID_DATE_FORMATS = [
    "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
    "%d/%m/%y", "%d-%m-%y", "%d.%m.%y",
    "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
    "%m/%d/%Y", "%m-%d-%Y",
    "%m/%d/%y", "%m-%d-%y",          # US 2-digit year: 04/27/19
    "%d %b %Y", "%d %B %Y",
    "%b %d, %Y", "%B %d, %Y",
    "%b %d %Y", "%B %d %Y",
]

LOW_CONFIDENCE_THRESHOLD = 0.7


# ── Validators ───────────────────────────────────────────────────────────────
def validate_date(date_str: Optional[str]) -> float:
    """
    Returns a multiplicative factor (0.0-1.0) reflecting how cleanly
    the string parses as a date.

      1.0  - parses with a strict format
      0.6  - matches a date-like regex but no exact format match
      0.2  - looks vaguely date-like
      0.0  - no signal at all
    """
    if not date_str:
        return 0.0
    s = date_str.strip()
    for fmt in VALID_DATE_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            # Sanity check — receipt dates shouldn't be in the far future or far past
            if 1990 <= dt.year <= 2099:
                return 1.0
            return 0.5
        except ValueError:
            continue
    if re.match(r"^\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}$", s):
        return 0.6
    if re.search(r"\d{1,2}[/\-.\s]\d{1,2}[/\-.\s]\d{2,4}", s):
        return 0.4
    return 0.2


def validate_currency(amount_str: Optional[str]) -> float:
    """
    Returns a multiplicative factor for currency-value plausibility.

      1.0  - parses as float in [0.01, 1,000,000]
      0.6  - parses but is suspiciously small or large
      0.0  - doesn't parse
    """
    if not amount_str:
        return 0.0
    s = str(amount_str).strip().replace(",", ".")
    try:
        val = float(s)
    except ValueError:
        return 0.0
    if 0.01 <= val <= 1_000_000:
        return 1.0
    if 0 < val < 100_000_000:
        return 0.6
    return 0.0


def validate_store_name(name: Optional[str]) -> float:
    """
    A store name should:
      - have at least 3 alpha chars
      - not be all uppercase noise like "AAAA"
      - not look like a date or a phone number
    """
    if not name:
        return 0.0
    s = name.strip()
    if len(s) < 3:
        return 0.3
    alpha = sum(1 for c in s if c.isalpha())
    if alpha < 3:
        return 0.3
    if re.match(r"^[\d\s/\-.:,()+]+$", s):
        return 0.2
    if alpha / len(s) < 0.3:   # mostly digits / symbols
        return 0.5
    return 1.0


# ── Combiner ─────────────────────────────────────────────────────────────────
def adjust_confidence(field: str, value: Optional[str],
                      raw_conf: float) -> float:
    """
    Combine raw OCR/heuristic confidence with field-specific validation.

    Args:
        field:    one of 'date', 'total_amount', 'price', 'store_name', or
                  any other (no special handling).
        value:    the extracted value (string or None).
        raw_conf: confidence from the extractor (already includes
                  OCR conf × heuristic factor).

    Returns:
        Final confidence in [0.0, 1.0], rounded to 3 decimals.
    """
    if value is None or str(value).strip() == "":
        return 0.0

    if field == "date":
        factor = validate_date(value)
    elif field in ("total_amount", "price"):
        factor = validate_currency(value)
    elif field == "store_name":
        factor = validate_store_name(value)
    else:
        factor = 1.0

    final = raw_conf * factor
    return round(min(max(final, 0.0), 1.0), 3)


def is_low_confidence(score: float) -> bool:
    """Convenience predicate matching the reviewer's suggested threshold."""
    return score < LOW_CONFIDENCE_THRESHOLD


def collect_low_confidence_flags(field_scores: dict) -> list:
    """
    Given a dict of {field_name: confidence_score}, return the list of
    field names that fall below the low-confidence threshold.
    """
    return [name for name, score in field_scores.items()
            if is_low_confidence(score)]
