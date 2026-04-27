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
    # Match DD/MM/YYYY, MM/DD/YY, DD.MM.YY etc. (allowing spaces for OCR noise)
    r"\b(\d{1,2}\s?[/\-.]\s?\d{1,2}\s?[/\-.]\s?\d{2,4})\b",
    # Match abbreviated months: 27-Apr-2026 or 27 Apr 26
    r"\b(\d{1,2}\s?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s?\d{2,4})\b",
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
    ("total purchase", 1),
    ("grand total", 100),
    ("total", 90),
    ("bal", 85),  # "BAL" or "BALANCE" common for Whole Foods
    ("amount due", 80),
    ("net amount", 75),
    ("payment due", 70),
    ("total purchase", 65),
    ("purchase total", 65),
    ("total due", 65),
    ("total tax invoice", 60),
    ("sum", 50),
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
    "save today", "special offer", "survey", "chance to win",
    "see back", "back of receipt", "for your chance",
    "win $", "win ₹", "win 1000", "win 5000",
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
        
        # Boost: Known major retailers
        known_retailers = ["walmart", "spar", "target", "costco", "kroger", "tesco"]
        if any(retailer in text_lower for retailer in known_retailers):
            position_bonus += 0.20
            # Clean up fuzzy OCR errors (e.g. "WALAMART" -> "Walmart")
            if "wal" in text_lower and "mart" in text_lower: text = "Walmart"
            if "spar" in text_lower: text = "SPAR"

        conf = min(1.0, line["confidence"] * 0.90 + position_bonus)
        return text, conf

    # Fallback — first line, low confidence
    return lines[0]["text"], lines[0]["confidence"] * 0.4


def extract_total(lines: List[Dict], max_value: float = 1_000_000) -> Optional[Tuple[str, float]]:
    """
    Find the total amount using a multi-pass approach:
    1. Keyword proximity and positioning
    2. Bbox-aware columnar lookup
    3. Mathematical item-sum fallback
    """
    raw_full = " ".join(l["text"] for l in lines)
    candidates = []
    
    # ── PASS 1: Keyword proximity with scoring ──
    for i, line in enumerate(lines):
        text_lower = line["text"].lower()
        # Skip noise lines
        if any(neg in text_lower for neg in
               ["discount", "rounding", "tax", "gst", "vat",
                "cash", "change", "tender", "tendered", "debit", "credit", "pay from"]):
            if not (text_lower.strip() == "total" or text_lower.startswith("total ")):
                continue

        for keyword, priority in TOTAL_KEYWORDS_PRIORITIZED:
            if keyword in text_lower:
                # Find number in this line or subsequent 2 lines
                for j in range(i, min(i + 3, len(lines))):
                    context = lines[j]["text"]
                    context_clean = context.replace(",", "")
                    vals = re.findall(DECIMAL_CURRENCY_RE, context_clean)
                    for v in vals:
                        try:
                            fval = float(v)
                            if 0.01 < fval < max_value:
                                if _looks_like_date_fragment(v, raw_full): continue
                                decimal_bonus = 0.05 if "." in v else 0.0
                                bottom_bonus = (j / len(lines)) * 0.15
                                keyword_score = priority / 100.0
                                score = (keyword_score + lines[j]["confidence"] * 0.7 + 
                                         decimal_bonus + bottom_bonus)
                                candidates.append((f"{fval:.2f}", score))
                        except ValueError:
                            continue
                break

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0]

    # ── PASS 2: Bbox-aware right-column lookup (Walmart style) ──
    total_kws = [kw for kw, _ in TOTAL_KEYWORDS_PRIORITIZED
                 if kw not in ("amount", "amt", "net", "sum", "balance")]
    filtered_lines = [l for l in lines if not any(x in l["text"].lower() for x in 
                      ["tax", "gst", "vat", "cash", "change", "tender"])]
    res = _find_keyword_value_with_bbox(filtered_lines, total_kws, raw_full)
    if res:
        val, conf = res
        return f"{val:.2f}", conf * 0.9

    return None
    return None


def calculate_sum_of_items(items: List[Dict]) -> float:
    """Calculate the mathematical sum of all extracted line items."""
    total = 0.0
    for item in items:
        try:
            p = item.get("price", "0")
            q = item.get("quantity", 1)
            total += float(p) * float(q)
        except (ValueError, TypeError):
            continue
    return total



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
        # Skip summary / header / footer
        # Strong match: line consists ALMOST entirely of skip keywords
        is_skip = False
        for k in ITEM_SKIP_KEYWORDS:
            if k in text_lower:
                # If the line IS the keyword (e.g. "TOTAL"), skip it
                if text_lower == k or text_lower.startswith(k + " ") or text_lower.endswith(" " + k):
                    is_skip = True
                    break
        if is_skip:
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
