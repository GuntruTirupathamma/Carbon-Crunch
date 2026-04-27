"""
Key information extraction from OCR output.

Strategy:
  - Date:        regex over multiple common formats (intl + IN + US + textual)
  - Store name:  first sufficiently-long, alpha-dominant line near the top
                 (positional heuristic + length filter to skip noise)
  - Total:       keyword-anchored search with priority + neighbor-line lookahead
                 + fallback to largest plausible currency value
  - Items:       lines that match "<text> <price>" pattern, excluding any
                 line containing summary/header/footer keywords

All extractors return (value, raw_confidence) tuples. The raw_confidence is
the OCR confidence multiplied by a heuristic factor reflecting how reliable
the extraction logic is for that field.
"""
from __future__ import annotations

import re
from typing import List, Dict, Optional, Tuple
import numpy as np


# ── Patterns ─────────────────────────────────────────────────────────────────
DATE_PATTERNS = [
    # 12/04/2026, 12-04-26, 12.04.2026, 13.12.01
    r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})\b",
    # 2026-04-12, 2026/04/12
    r"\b(\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2})\b",
    # 12 Apr 2026, 12 April 2026
    r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"[a-z]*\s+\d{2,4})\b",
    # April 12, 2026
    r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+"
    r"\d{1,2},?\s+\d{2,4})\b",
]

# Currency pattern — captures the numeric value (with optional symbol prefix)
CURRENCY_RE = re.compile(
    r"(?:(?:₹|Rs\.?|RM|INR|\$|USD|EUR|£|GBP)\s*)?"
    r"(\d{1,3}(?:[,]\d{3})+(?:\.\d{1,2})?|\d+\.\d{1,2}|\d+)"
)

# Decimal-only currency (more reliable signal — receipts almost always show
# totals with 2 decimal places)
DECIMAL_CURRENCY_RE = re.compile(
    r"(?:(?:₹|Rs\.?|RM|INR|\$|USD|EUR|£|GBP)\s*)?"
    r"(\d{1,3}(?:[,]\d{3})*\.\d{2}|\d+\.\d{2})"
)

# Strict price line: "<words> <price-with-decimals>"
PRICE_LINE_RE = re.compile(
    r"^(.+?)\s+(?:₹|Rs\.?|RM|\$)?\s*(\d+(?:[,]\d{3})*\.\d{2}|\d+\.\d{2})\s*[A-Z]?\s*$"
)

TOTAL_KEYWORDS_PRIORITIZED = [
    ("grand total", 0),
    ("grandtotal", 0),
    ("net total", 1),
    ("nett total", 1),
    ("total amount", 1),
    ("amount due", 1),
    ("balance due", 1),
    ("amount payable", 1),
    ("total", 2),
    ("amount", 3),
    ("amt", 3),
    ("net", 3),
    ("sum", 4),
    ("balance", 4),
]

# Lines containing these are NEVER items
ITEM_SKIP_KEYWORDS = [
    "total", "subtotal", "sub total", "sub-total",
    "tax", "gst", "vat", "service",
    "discount", "rounding", "round off",
    "change", "cash", "card", "tendered", "tender",
    "payment", "paid", "tip",
    "thank", "visit", "welcome",
    "invoice", "receipt", "bill no", "bill #",
    "phone", "tel:", "tel ", "email", "address",
    "table", "guest", "server", "cashier",
    "date", "time", "no.",
    "qty", "quantity", "unit",
]

# Lines containing these are usually header/footer noise — skip when finding
# the store name
STORE_NAME_NOISE = [
    "always low",  # "Always Low Prices" tagline
    "supercenter", "open ", "manager",
    "tel:", "phone", "address",
    "thank", "welcome", "receipt",
    "invoice", "bill no", "bill #", "order #",
    # marketing / competition text at top of many receipts
    "see back", "back of", "chance to win", "win $", "win ₹",
    "your chance", "sweepstakes", "survey", "coupon",
    "save today", "special offer",
]


# ── Helpers ──────────────────────────────────────────────────────────────────
def _parse_amount(s: str) -> Optional[float]:
    """Convert '1,234.56' or '1.234,56' (eu) → 1234.56 float."""
    s = str(s).strip().replace(" ", "")
    if not s:
        return None
    # Heuristic: if both . and , present, the rightmost is the decimal
    if "." in s and "," in s:
        if s.rfind(".") > s.rfind(","):
            s = s.replace(",", "")
        else:
            s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        # If exactly groups of 3 digits → thousands separator
        if re.match(r"^\d{1,3}(,\d{3})+$", s):
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _alpha_ratio(s: str) -> float:
    if not s:
        return 0.0
    alpha = sum(1 for c in s if c.isalpha())
    return alpha / max(len(s), 1)


def _looks_like_date_fragment(value: str, all_text: str) -> bool:
    """
    True if `value` is just a date fragment (e.g. '13.12' is part of '13.12.01').
    Prevents date prefixes from being mistaken for currency totals.
    """
    if not value:
        return False
    # Search for date-like pattern containing this value
    # e.g. value='13.12' would match in '13.12.01' or '13.12.2026'
    pattern = re.escape(value) + r"[/\-.]\d{2,4}"
    return bool(re.search(pattern, all_text))


def _bbox_y_range(bbox) -> Tuple[float, float]:
    """Return (y_top, y_bottom) of a bbox."""
    if not bbox:
        return 0.0, 0.0
    ys = [pt[1] for pt in bbox]
    return min(ys), max(ys)


def _vertically_overlap(a: Dict, b: Dict, min_overlap: float = 0.4) -> bool:
    """
    True if two detections overlap vertically by at least `min_overlap`
    fraction of the smaller line's height. This is more reliable than
    y_center proximity for tightly-spaced rows.
    """
    a_top, a_bot = _bbox_y_range(a.get("bbox"))
    b_top, b_bot = _bbox_y_range(b.get("bbox"))
    if a_bot <= a_top or b_bot <= b_top:
        return False
    inter = max(0.0, min(a_bot, b_bot) - max(a_top, b_top))
    smaller_h = min(a_bot - a_top, b_bot - b_top)
    return (inter / smaller_h) >= min_overlap


def _merge_lines_into_rows(lines: List[Dict]) -> List[Dict]:
    """
    Receipts are columnar layouts — OCR often splits 'BANANAS  0.20' into two
    separate detections. Merge detections whose bounding boxes share enough
    vertical overlap (≥40% of the smaller line's height).

    Returns a list of {text, confidence, y_center} where text is the joined
    row content, sorted left-to-right.
    """
    if not lines:
        return []
    sorted_lines = sorted(lines, key=lambda l: l["y_center"])
    rows: List[List[Dict]] = []
    for line in sorted_lines:
        # Try to attach to the LAST row only if bboxes vertically overlap
        attached = False
        if rows:
            # Check overlap with any element of the last row (some elements may
            # be slightly higher/lower than others)
            for other in rows[-1]:
                if _vertically_overlap(line, other):
                    rows[-1].append(line)
                    attached = True
                    break
        if not attached:
            rows.append([line])

    merged = []
    for row in rows:
        row_sorted = sorted(row, key=lambda l: l["bbox"][0][0] if l.get("bbox") else 0)
        text = " ".join(l["text"] for l in row_sorted)
        conf = float(np.mean([l["confidence"] for l in row_sorted]))
        y    = float(np.mean([l["y_center"] for l in row_sorted]))
        merged.append({"text": text, "confidence": conf, "y_center": y})
    return merged


# ── Extractors ───────────────────────────────────────────────────────────────
def extract_date(lines: List[Dict]) -> Optional[Tuple[str, float]]:
    """
    Returns (date_string, confidence) or None.
    Validates that the matched text actually parses as a calendar date
    (not a price like '6-10.00' that superficially matches the date regex).
    """
    from src.confidence import validate_date as _vd
    for line in lines:
        for pattern in DATE_PATTERNS:
            m = re.search(pattern, line["text"], re.IGNORECASE)
            if m:
                candidate = m.group(1)
                # Reject if it looks like a currency value (contains decimal .NN at end)
                if re.search(r"\.\d{2}$", candidate):
                    continue
                # Reject if validate_date gives very low score (probably not a real date)
                if _vd(candidate) < 0.4:
                    continue
                return candidate, line["confidence"] * 0.95
    return None


def extract_store_name(lines: List[Dict]) -> Tuple[str, float]:
    """
    Heuristic: store name is one of the first few alpha-dominant lines,
    long enough to be a real name, not a tagline or address line.
    Extended to scan up to 10 lines to skip common receipt header marketing copy.
    """
    if not lines:
        return "Unknown", 0.0

    # Expand search window — some receipts have 2-3 header lines before the brand
    for idx, line in enumerate(lines[:10]):
        text = line["text"].strip()
        text_lower = text.lower()

        # Quality filters
        if len(text) < 3:
            continue
        if _alpha_ratio(text) < 0.5:
            continue
        if any(noise in text_lower for noise in STORE_NAME_NOISE):
            continue
        # Skip if it looks like a date / phone / number
        if re.match(r"^[\d\s/\-.:,()+]+$", text):
            continue
        # Skip very short single-letter combinations
        if sum(1 for c in text if c.isalpha()) < 3:
            continue
        # Skip if it reads like a full sentence (store names are ≤ 6 words usually)
        word_count = len(text.split())
        if word_count > 7:
            continue
        # Skip obvious contest / marketing lines
        if re.search(r"\$\d{3,}|#\s*\d{6,}", text):
            continue

        position_bonus = 0.10 if idx == 0 else (0.07 if idx == 1 else 0.04 if idx <= 3 else 0.0)
        # Boost: ALL CAPS short words at the top are usually brand names
        if text.isupper() and 3 <= len(text) <= 30:
            position_bonus += 0.05
        conf = min(1.0, line["confidence"] * 0.90 + position_bonus)
        return text, conf

    # Fallback — first line, low confidence
    return lines[0]["text"], lines[0]["confidence"] * 0.4


def extract_total(lines: List[Dict]) -> Optional[Tuple[str, float]]:
    """
    Find the total amount.

    Priority:
      1. "Grand Total" keyword line (merged row)
      2. "Net Total / Amount Due / Balance Due"
      3. "Total" (alone) — checked both in merged text AND via bbox right-column lookup
      4. Other total-like keywords
      5. Fallback — most-frequent or bottom-half largest plausible currency value
    """
    # Use row-merged lines so 'TOTAL' (left col) and '5.11' (right col) end up
    # together, even when EasyOCR returned them as separate detections.
    merged = _merge_lines_into_rows(lines)
    full_text = " ".join(l["text"] for l in merged)
    raw_full  = " ".join(l["text"] for l in lines)

    # ── PASS 1: scan merged rows (handles receipts where EasyOCR already joins columns) ──
    candidates = []
    for i, line in enumerate(merged):
        text_lower = line["text"].lower()

        # Skip lines that are clearly NOT totals
        if any(neg in text_lower for neg in
               ["subtotal", "sub total", "sub-total",
                "discount", "rounding", "tax", "gst", "vat",
                "cash", "change", "tender", "tendered"]):
            continue

        for keyword, priority in TOTAL_KEYWORDS_PRIORITIZED:
            if keyword in text_lower:
                # Strip UPC/barcodes before searching for the price
                clean_text = _UPC_RE.sub(" ", line["text"])
                matches = list(DECIMAL_CURRENCY_RE.finditer(clean_text))
                price_str = None
                if matches:
                    price_str = matches[-1].group(1)
                elif i + 1 < len(merged):
                    # Only look at next line if it doesn't look like cash/change
                    next_text_lower = merged[i + 1]["text"].lower()
                    if not any(x in next_text_lower for x in
                               ["cash", "change", "tender", "tax", "gst"]):
                        next_clean = _UPC_RE.sub(" ", merged[i + 1]["text"])
                        next_decimals = list(DECIMAL_CURRENCY_RE.finditer(next_clean))
                        if next_decimals:
                            price_str = next_decimals[-1].group(1)

                if price_str:
                    parsed = _parse_amount(price_str)
                    if _looks_like_date_fragment(price_str, raw_full):
                        continue
                    if parsed is not None and 0.01 <= parsed < 1_000_000:
                        decimal_bonus = 0.05 if "." in price_str else -0.10
                        candidates.append({
                            "value": f"{parsed:.2f}",
                            "confidence": min(1.0,
                                line["confidence"] * 0.92 + decimal_bonus),
                            "priority": priority,
                        })
                        break

    if candidates:
        candidates.sort(key=lambda c: (c["priority"], -c["confidence"]))
        best = candidates[0]
        return best["value"], best["confidence"]

    # ── PASS 2: bbox-aware right-column lookup (handles split-column receipts) ──
    # This catches Walmart-style receipts where "TOTAL" (left) and "144.02" (right)
    # are separate OCR detections that vertical-overlap but aren't text-merged.
    total_kws = [kw for kw, _ in TOTAL_KEYWORDS_PRIORITIZED
                 if kw not in ("amount", "amt", "net", "sum", "balance")]
    _filtered_lines = [
        l for l in lines
        if not any(x in l["text"].lower()
                   for x in ["subtotal", "sub total", "tax", "gst", "vat",
                              "discount", "cash", "change", "tender"])
    ]
    res = _find_keyword_value_with_bbox(_filtered_lines, total_kws, raw_full,
                                        min_overlap=0.2)
    if res:
        val, conf = res
        return f"{val:.2f}", min(1.0, conf * 0.90)

    # ── PASS 3: simple y-center proximity scan (most robust — no bbox required) ──
    # For each raw line containing a total keyword, find the nearest decimal number
    # within 80px y-distance.  Works even when OCR returns no bbox data.
    TOTAL_KW_SET = {kw for kw, _ in TOTAL_KEYWORDS_PRIORITIZED
                    if kw not in ("amount", "amt", "net", "sum", "balance")}
    SKIP_KW_SET  = {"subtotal", "sub total", "sub-total", "tax", "gst", "vat",
                    "discount", "cash", "change", "tender", "tendered", "debit",
                    "credit"}
    _pass3_candidates = []
    for kw_line in lines:
        tl = kw_line["text"].lower()
        if any(s in tl for s in SKIP_KW_SET):
            continue
        if not any(kw in tl for kw in TOTAL_KW_SET):
            continue
        kw_y = kw_line["y_center"]

        # Search for a decimal value in this line first
        clean = _UPC_RE.sub(" ", kw_line["text"])
        m = DECIMAL_CURRENCY_RE.search(clean)
        if m:
            price_str = m.group(1)
            if not _looks_like_date_fragment(price_str, raw_full):
                parsed = _parse_amount(price_str)
                if parsed and 0.01 <= parsed < 1_000_000:
                    _pass3_candidates.append((0, parsed, kw_line["confidence"]))
            continue

        # Not in same line — search nearby lines by y-center distance
        for other in lines:
            if other is kw_line:
                continue
            y_dist = abs(other["y_center"] - kw_y)
            if y_dist > 80:
                continue
            ot = other["text"].lower()
            if any(s in ot for s in SKIP_KW_SET):
                continue
            clean2 = _UPC_RE.sub(" ", other["text"])
            m2 = DECIMAL_CURRENCY_RE.search(clean2)
            if not m2:
                continue
            price_str = m2.group(1)
            if _looks_like_date_fragment(price_str, raw_full):
                continue
            parsed = _parse_amount(price_str)
            if parsed and 0.01 <= parsed < 1_000_000:
                _pass3_candidates.append((y_dist, parsed, other["confidence"]))

    if _pass3_candidates:
        # Prefer closest to keyword line, then highest value
        _pass3_candidates.sort(key=lambda x: (x[0], -x[1]))
        _, val, conf = _pass3_candidates[0]
        return f"{val:.2f}", conf * 0.80

    # ── Fallback: collect plausible decimal amounts from merged rows ─────
    all_amounts = []
    for line in merged:
        text = _UPC_RE.sub(" ", line["text"]).strip()   # strip barcodes first
        text_lower = text.lower()
        # Skip cash-tender / change lines — they're larger than the total
        if any(x in text_lower for x in ["cash", "change", "tender", "credit", "debit"]):
            continue
        if re.search(r"[A-Z]{2,}\d{4,}", text):  continue   # ITC5679...
        if len(re.findall(r"\d", text)) > 12:    continue   # transaction ids

        for m in DECIMAL_CURRENCY_RE.finditer(text):
            raw  = m.group(1)
            parsed = _parse_amount(raw)
            if parsed is None or not (0.01 <= parsed <= 100_000):
                continue
            if _looks_like_date_fragment(raw, raw_full):
                continue
            all_amounts.append((parsed, line["confidence"], line["y_center"]))

    if all_amounts:
        from collections import Counter
        counter = Counter(round(v, 2) for v, _, _ in all_amounts)
        most_common_val, freq = counter.most_common(1)[0]
        if freq >= 2:
            conf = max(c for v, c, _ in all_amounts if round(v, 2) == most_common_val)
            return f"{most_common_val:.2f}", conf * 0.7

        # Largest plausible value in the bottom half (totals are at the bottom)
        n = len(merged)
        if n > 1:
            mid_y = merged[n // 2]["y_center"]
            bottom = [(v, c) for v, c, y in all_amounts if y >= mid_y]
            if bottom:
                bottom.sort(key=lambda x: -x[0])
                return f"{bottom[0][0]:.2f}", bottom[0][1] * 0.6

        # Final fallback
        all_amounts.sort(key=lambda x: -x[0])
        v, c, _ = all_amounts[0]
        return f"{v:.2f}", c * 0.45

    return None



def _find_value_after_keyword(text: str, keywords: list, full_text: str,
                               max_value: float = 1_000_000) -> Optional[float]:
    """
    Find the FIRST decimal price that appears immediately after one of the
    given keywords in `text`. Avoids picking up later prices that belong to
    different fields (e.g., picking TAX value when looking for SUBTOTAL).
    """
    text_lower = text.lower()
    earliest_pos = None
    matched_keyword = None
    for kw in keywords:
        pos = text_lower.find(kw)
        if pos != -1 and (earliest_pos is None or pos < earliest_pos):
            earliest_pos = pos
            matched_keyword = kw
    if earliest_pos is None:
        return None
    # Search the region right after the keyword
    after = text[earliest_pos + len(matched_keyword):]
    m = DECIMAL_CURRENCY_RE.search(after)
    if not m:
        return None
    price_str = m.group(1)
    if _looks_like_date_fragment(price_str, full_text):
        return None
    parsed = _parse_amount(price_str)
    if parsed and 0.01 <= parsed < max_value:
        return parsed
    return None


def _find_keyword_value_with_bbox(lines: List[Dict], keywords: list,
                                   full_text: str,
                                   max_value: float = 1_000_000,
                                   min_overlap: float = 0.4,
                                   ) -> Optional[Tuple[float, float]]:
    """
    For each line containing a keyword, find a value in the SAME ROW
    by matching bbox y-ranges (right-column lookup).

    Returns (value, confidence) or None.
    """
    # Find lines containing any keyword (left column)
    for kw_line in lines:
        text_lower = kw_line["text"].lower()
        kw_match = None
        for kw in keywords:
            if kw in text_lower:
                kw_match = kw
                break
        if not kw_match:
            continue
        # Skip false positives
        if "invoice" in text_lower or "exempt" in text_lower:
            continue

        # First check if there's a decimal value already in this same line
        text = kw_line["text"]
        kw_pos = text_lower.find(kw_match)
        after = text[kw_pos + len(kw_match):]
        m = DECIMAL_CURRENCY_RE.search(after)
        if m:
            price_str = m.group(1)
            if not _looks_like_date_fragment(price_str, full_text):
                parsed = _parse_amount(price_str)
                if parsed and 0.01 <= parsed < max_value:
                    return parsed, kw_line["confidence"]

        # Look at OTHER lines that vertically overlap, are to the RIGHT,
        # and pick the one with y_center CLOSEST to the keyword (handles
        # tightly-packed rows where multiple values overlap).
        kw_x_end = max(pt[0] for pt in kw_line["bbox"]) if kw_line.get("bbox") else 0
        kw_y     = kw_line["y_center"]
        candidates = []
        for other in lines:
            if other is kw_line:
                continue
            if not _vertically_overlap(kw_line, other, min_overlap=min_overlap):
                continue
            other_x_start = min(pt[0] for pt in other["bbox"]) if other.get("bbox") else 0
            if other_x_start < kw_x_end:
                continue
            m = DECIMAL_CURRENCY_RE.search(other["text"])
            if not m:
                continue
            price_str = m.group(1)
            if _looks_like_date_fragment(price_str, full_text):
                continue
            parsed = _parse_amount(price_str)
            if parsed is None or not (0.01 <= parsed < max_value):
                continue
            y_dist = abs(other["y_center"] - kw_y)
            candidates.append((y_dist, parsed, other["confidence"]))

        if candidates:
            candidates.sort(key=lambda c: c[0])
            _, val, conf = candidates[0]
            return val, conf
    return None


def extract_subtotal(lines: List[Dict]) -> Optional[Tuple[str, float]]:
    """Find subtotal using bbox-aware right-column lookup."""
    full_text = " ".join(l["text"] for l in lines)
    res = _find_keyword_value_with_bbox(
        lines, ["subtotal", "sub total", "sub-total"], full_text)
    if res:
        val, conf = res
        return f"{val:.2f}", conf * 0.92
    return None


def extract_tax(lines: List[Dict]) -> Optional[Tuple[str, float]]:
    """Find tax / GST / VAT using bbox-aware right-column lookup."""
    full_text = " ".join(l["text"] for l in lines)
    # Order matters — match longer keywords first to avoid 'tax' matching inside 'service tax'
    keywords = ["service tax", "cgst", "sgst", "igst", "gst", "vat", "tax"]
    res = _find_keyword_value_with_bbox(lines, keywords, full_text,
                                          max_value=100_000)
    if res:
        val, conf = res
        return f"{val:.2f}", conf * 0.90
    return None


CATEGORY_KEYWORDS = {
    "Groceries":   ["walmart", "supermarket", "supercenter", "grocery", "kroger",
                    "trader", "whole foods", "safeway", "tesco", "sainsbury",
                    "big bazaar", "dmart", "reliance fresh", "more", "spar",
                    "lidl", "aldi", "carrefour", "costco"],
    "Dining":      ["restaurant", "cafe", "coffee", "starbucks", "pizza",
                    "mcdonald", "kfc", "subway", "burger", "barbeque", "bistro",
                    "diner", "kitchen", "grill", "tea", "chai", "bakery",
                    "domino", "chipotle"],
    "Transport":   ["uber", "lyft", "ola", "taxi", "rapido", "metro",
                    "indian oil", "iocl", "bpcl", "hpcl", "shell", "fuel",
                    "petrol", "diesel", "gas station", "parking", "toll"],
    "Office":      ["office depot", "staples", "stationery", "supplies",
                    "printing", "xerox", "ricoh"],
    "Utilities":   ["electric", "electricity", "water", "internet", "telecom",
                    "airtel", "jio", "vodafone", "bsnl", "vi ", "verizon",
                    "comcast", "at&t"],
    "Travel":      ["airline", "indigo", "spicejet", "vistara", "delta",
                    "united", "emirates", "hotel", "marriott", "hilton",
                    "hyatt", "oyo", "airbnb", "irctc", "amtrak"],
    "Healthcare":  ["pharmacy", "clinic", "hospital", "medical", "apollo",
                    "fortis", "max healthcare", "cvs", "walgreens"],
    "Software":    ["software", "license", "subscription", "saas", "github",
                    "aws", "azure", "google cloud", "microsoft", "adobe"],
    "Retail":      ["amazon", "flipkart", "myntra", "best buy", "target",
                    "ikea", "zara", "h&m", "nike"],
}


def detect_category(store_name: str, full_text: str) -> Tuple[str, float]:
    """
    Classify the receipt into a spending category based on store name + text.
    Returns (category, confidence) — confidence reflects how strong the match is.
    """
    haystack = f"{store_name} {full_text}".lower()
    scores: Dict[str, int] = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in haystack)
        if hits > 0:
            scores[cat] = hits
    if not scores:
        return "Uncategorized", 0.0
    best = max(scores.items(), key=lambda x: x[1])
    cat, hits = best
    # Confidence scales with hits, capped at 0.95
    conf = min(0.95, 0.55 + 0.15 * hits)
    return cat, conf


def detect_currency(lines: List[Dict]) -> Tuple[str, str]:
    """
    Detect the currency used.
    Returns (symbol, code) e.g. ('₹', 'INR'), ('$', 'USD'), ('€', 'EUR').
    """
    full_text = " ".join(l["text"] for l in lines)
    full_lower = full_text.lower()

    # Symbols (most reliable)
    if "₹" in full_text:                                         return "₹", "INR"
    if "€" in full_text:                                         return "€", "EUR"
    if "£" in full_text:                                         return "£", "GBP"
    if "¥" in full_text:                                         return "¥", "JPY"

    # Codes
    if re.search(r"\bRM\b|ringgit", full_text):                  return "RM", "MYR"
    if re.search(r"\bINR\b|rupee", full_lower):                  return "₹", "INR"
    if re.search(r"\bRs\.?\b", full_text):                       return "₹", "INR"
    if re.search(r"\bUSD\b|us\s*dollar", full_lower):            return "$", "USD"
    if re.search(r"\bEUR\b|euro", full_lower):                   return "€", "EUR"
    if re.search(r"\bGBP\b|pound", full_lower):                  return "£", "GBP"
    if re.search(r"\bSGD\b|singapore\s*dollar", full_lower):     return "S$", "SGD"

    # Country-specific store hints
    if re.search(r"walmart|target|costco|cvs|walgreens", full_lower):  return "$", "USD"
    if re.search(r"big\s*bazaar|reliance|dmart|tata", full_lower):     return "₹", "INR"

    # Default: dollar (most international)
    if "$" in full_text:                                         return "$", "USD"
    return "$", "USD"


# UPC / barcode pattern — 12-digit retail barcodes and internal store codes
_UPC_RE = re.compile(r"\b\d{10,}\b")
# Single trailing tax/status codes on Walmart receipts (e.g. " F", " N", " X", " 0")
_TRAIL_CODE_RE = re.compile(r"\s+[A-Z0]\s*$")


def _strip_receipt_codes(text: str) -> str:
    """
    Remove UPC barcodes and trailing tax flag characters from a receipt line
    so that item descriptions and prices can be extracted cleanly.

    Example:
        'TATER TOTS  003131200036 F   2.98 0'
        → 'TATER TOTS 2.98'
    """
    # Remove long numeric codes (UPC, SKU)
    text = _UPC_RE.sub(" ", text)
    # Remove trailing single-letter tax/status flags
    text = _TRAIL_CODE_RE.sub("", text)
    # Collapse whitespace
    return re.sub(r"\s{2,}", " ", text).strip()


def extract_items(lines: List[Dict]) -> List[Dict]:
    """
    Extract line-items using row-merged OCR output.

    Receipts are columnar — OCR often splits 'BANANAS' (left) and '0.20' (right)
    into two separate detections. We first merge by y-coordinate, THEN look for
    the '<text> ... <decimal-price>' pattern.

    UPC codes (12+ digits) are stripped *before* pattern matching so that
    Walmart / supermarket receipts are not silently dropped.
    """
    merged = _merge_lines_into_rows(lines)
    full_text = " ".join(l["text"] for l in merged)
    items = []

    for line in merged:
        raw_text = line["text"].strip()
        text_lower = raw_text.lower()

        # Skip summary / header / footer  (check BOTH raw and stripped text)
        if any(k in text_lower for k in ITEM_SKIP_KEYWORDS):
            continue

        # Strip UPC / store codes before any further processing
        text = _strip_receipt_codes(raw_text)
        if not text:
            continue

        # Re-check skip keywords on the cleaned text
        if any(k in text.lower() for k in ITEM_SKIP_KEYWORDS):
            continue

        # Skip rows that are ONLY codes/numbers (e.g. barcode-only lines)
        if re.search(r"[A-Z]{2,}\d{4,}", text):
            continue

        # Looser pattern: row contains a decimal price somewhere
        price_matches = list(DECIMAL_CURRENCY_RE.finditer(text))
        if not price_matches:
            continue

        # Take the LAST decimal value as the price (rightmost = price column)
        last = price_matches[-1]
        price_raw = last.group(1)
        if _looks_like_date_fragment(price_raw, full_text):
            continue
        price = _parse_amount(price_raw)
        if price is None or not (0.01 <= price <= 10_000):
            continue

        # Name is everything BEFORE the price
        name = text[:last.start()].strip().rstrip(":-").strip()

        # Quality filters on the name
        alpha_count = sum(1 for c in name if c.isalpha())
        if alpha_count < 3:                    continue
        if len(name) > 80:                     continue
        if _alpha_ratio(name) < 0.25:          continue  # relaxed for codes still present
        if name.count("#") >= 2:               continue
        if name.count(":") >= 2:               continue

        # Try to extract quantity from the name (e.g. "2 x Milk" or "Milk 2")
        qty = 1
        qty_match = re.match(r"^\s*(\d{1,3})\s*[xX]?\s+(.+)$", name)
        if qty_match:
            try:
                possible_qty = int(qty_match.group(1))
                if 1 <= possible_qty <= 99:   # plausible quantity
                    qty = possible_qty
                    name = qty_match.group(2).strip()
            except ValueError:
                pass

        items.append({
            "description": name,
            "quantity":    qty,
            "price":       f"{price:.2f}",
            "confidence":  line["confidence"] * 0.78,
        })

    return items
