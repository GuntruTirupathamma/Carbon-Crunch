"""
Carbon Crunch — AI Receipt Intelligence
========================================
Self-contained Streamlit app (all OCR + extraction logic merged inline so
the Streamlit Cloud deploy doesn't depend on `src/` package imports).

Run with:   streamlit run app.py
"""
from __future__ import annotations

import io
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st


# ═══════════════════════════════════════════════════════════════════════════════
# 0. PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Carbon Crunch · AI Receipt Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CARBON_KG_PER_UNIT = 0.041   # rough industry-average kg CO₂e per ₹/$


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PREPROCESSING (merged from src/preprocess.py)
# ═══════════════════════════════════════════════════════════════════════════════
def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def estimate_blur(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def denoise(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.fastNlMeansDenoising(image, None, h=10,
                                         templateWindowSize=7, searchWindowSize=21)
    return cv2.fastNlMeansDenoisingColored(image, None, h=10, hColor=10,
                                            templateWindowSize=7, searchWindowSize=21)


def deskew(image: np.ndarray) -> Tuple[np.ndarray, float]:
    gray = to_grayscale(image)
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
    if abs(angle) < 0.5:
        return image, 0.0
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return (cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE),
            float(angle))


def resize_for_ocr(image: np.ndarray, target_height: int = 1200,
                    max_height: int = 1800) -> np.ndarray:
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


# ═══════════════════════════════════════════════════════════════════════════════
# 2. OCR (merged from src/ocr.py)
# ═══════════════════════════════════════════════════════════════════════════════
_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(['en'], gpu=False, verbose=False,
                                  download_enabled=True)
    return _reader


def run_ocr(image: np.ndarray) -> List[Dict]:
    reader = _get_reader()
    try:
        raw = reader.readtext(image, detail=1, paragraph=False, batch_size=8)
    except Exception:
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
    return "\n".join(l["text"] for l in lines)


def get_average_confidence(lines: List[Dict]) -> float:
    if not lines:
        return 0.0
    return float(np.mean([l["confidence"] for l in lines]))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CONFIDENCE (merged from src/confidence.py)
# ═══════════════════════════════════════════════════════════════════════════════
VALID_DATE_FORMATS = [
    "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
    "%d/%m/%y", "%d-%m-%y", "%d.%m.%y",
    "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
    "%m/%d/%Y", "%m-%d-%Y",
    "%m/%d/%y", "%m-%d-%y",
    "%d %b %Y", "%d %B %Y",
    "%b %d, %Y", "%B %d, %Y",
    "%b %d %Y", "%B %d %Y",
]
LOW_CONFIDENCE_THRESHOLD = 0.7


def validate_date(date_str: Optional[str]) -> float:
    if not date_str: return 0.0
    s = date_str.strip()
    for fmt in VALID_DATE_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            if 1990 <= dt.year <= 2099: return 1.0
            return 0.5
        except ValueError:
            continue
    if re.match(r"^\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}$", s): return 0.6
    if re.search(r"\d{1,2}[/\-.\s]\d{1,2}[/\-.\s]\d{2,4}", s): return 0.4
    return 0.2


def validate_currency(amount_str: Optional[str]) -> float:
    if not amount_str: return 0.0
    s = str(amount_str).strip().replace(",", ".")
    try:
        val = float(s)
    except ValueError:
        return 0.0
    if 0.01 <= val <= 1_000_000: return 1.0
    if 0 < val < 100_000_000: return 0.6
    return 0.0


def validate_store_name(name: Optional[str]) -> float:
    if not name: return 0.0
    s = name.strip()
    if len(s) < 3: return 0.3
    alpha = sum(1 for c in s if c.isalpha())
    if alpha < 3: return 0.3
    if re.match(r"^[\d\s/\-.:,()+]+$", s): return 0.2
    if alpha / len(s) < 0.3: return 0.5
    return 1.0


def adjust_confidence(field: str, value: Optional[str], raw_conf: float) -> float:
    if value is None or str(value).strip() == "": return 0.0
    if field == "date":             factor = validate_date(value)
    elif field in ("total_amount", "price"): factor = validate_currency(value)
    elif field == "store_name":     factor = validate_store_name(value)
    else:                            factor = 1.0
    return round(min(max(raw_conf * factor, 0.0), 1.0), 3)


def collect_low_confidence_flags(field_scores: dict) -> list:
    return [n for n, s in field_scores.items() if s < LOW_CONFIDENCE_THRESHOLD]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EXTRACTOR (merged from src/extractor.py)
# ═══════════════════════════════════════════════════════════════════════════════
DATE_PATTERNS = [
    r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})\b",
    r"\b(\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2})\b",
    r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b",
    r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})\b",
]
DECIMAL_CURRENCY_RE = re.compile(
    r"(?:(?:₹|Rs\.?|RM|INR|\$|USD|EUR|£|GBP)\s*)?"
    r"(\d{1,3}(?:[,]\d{3})*\.\d{2}|\d+\.\d{2})"
)
TOTAL_KEYWORDS_PRIORITIZED = [
    ("grand total", 0), ("grandtotal", 0),
    ("net total", 1), ("nett total", 1),
    ("total amount", 1), ("amount due", 1), ("balance due", 1), ("amount payable", 1),
    ("total", 2),
    ("amount", 3), ("amt", 3), ("net", 3),
    ("sum", 4), ("balance", 4),
]
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
STORE_NAME_NOISE = [
    "always low", "supercenter", "open ", "manager",
    "tel:", "phone", "address",
    "thank", "welcome", "receipt",
    "invoice", "bill no", "bill #", "order #",
]


def _parse_amount(s: str) -> Optional[float]:
    s = str(s).strip().replace(" ", "")
    if not s: return None
    if "." in s and "," in s:
        if s.rfind(".") > s.rfind(","):
            s = s.replace(",", "")
        else:
            s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        if re.match(r"^\d{1,3}(,\d{3})+$", s):
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _alpha_ratio(s: str) -> float:
    if not s: return 0.0
    return sum(1 for c in s if c.isalpha()) / max(len(s), 1)


def _looks_like_date_fragment(value: str, all_text: str) -> bool:
    if not value: return False
    pattern = re.escape(value) + r"[/\-.]\d{2,4}"
    return bool(re.search(pattern, all_text))


def _vertically_overlap(a: Dict, b: Dict, min_overlap: float = 0.4) -> bool:
    a_ys = [pt[1] for pt in a.get("bbox", [])]
    b_ys = [pt[1] for pt in b.get("bbox", [])]
    if not a_ys or not b_ys: return False
    a_top, a_bot = min(a_ys), max(a_ys)
    b_top, b_bot = min(b_ys), max(b_ys)
    if a_bot <= a_top or b_bot <= b_top: return False
    inter = max(0.0, min(a_bot, b_bot) - max(a_top, b_top))
    smaller_h = min(a_bot - a_top, b_bot - b_top)
    return (inter / smaller_h) >= min_overlap


def _merge_lines_into_rows(lines: List[Dict]) -> List[Dict]:
    if not lines: return []
    sorted_lines = sorted(lines, key=lambda l: l["y_center"])
    rows = []
    for line in sorted_lines:
        attached = False
        if rows:
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
        merged.append({
            "text": " ".join(l["text"] for l in row_sorted),
            "confidence": float(np.mean([l["confidence"] for l in row_sorted])),
            "y_center": float(np.mean([l["y_center"] for l in row_sorted])),
        })
    return merged


def extract_date(lines: List[Dict]) -> Optional[Tuple[str, float]]:
    for line in lines:
        for pattern in DATE_PATTERNS:
            m = re.search(pattern, line["text"], re.IGNORECASE)
            if m:
                return m.group(1), line["confidence"] * 0.95
    return None


def extract_store_name(lines: List[Dict]) -> Tuple[str, float]:
    if not lines: return "Unknown", 0.0
    for idx, line in enumerate(lines[:6]):
        text = line["text"].strip()
        text_lower = text.lower()
        if len(text) < 3: continue
        if _alpha_ratio(text) < 0.5: continue
        if any(noise in text_lower for noise in STORE_NAME_NOISE): continue
        if re.match(r"^[\d\s/\-.:,()+]+$", text): continue
        if sum(1 for c in text if c.isalpha()) < 3: continue
        position_bonus = 0.10 if idx == 0 else (0.05 if idx == 1 else 0.0)
        if text.isupper() and 3 <= len(text) <= 30:
            position_bonus += 0.05
        return text, min(1.0, line["confidence"] * 0.90 + position_bonus)
    return lines[0]["text"], lines[0]["confidence"] * 0.4


def extract_total(lines: List[Dict]) -> Optional[Tuple[str, float]]:
    merged = _merge_lines_into_rows(lines)
    full_text = " ".join(l["text"] for l in merged)
    candidates = []
    for i, line in enumerate(merged):
        text_lower = line["text"].lower()
        if any(neg in text_lower for neg in
                ["subtotal", "sub total", "sub-total", "discount",
                 "rounding", "tax", "gst", "vat"]):
            continue
        for keyword, priority in TOTAL_KEYWORDS_PRIORITIZED:
            if keyword in text_lower:
                matches = list(DECIMAL_CURRENCY_RE.finditer(line["text"]))
                price_str = None
                if matches:
                    price_str = matches[-1].group(1)
                elif i + 1 < len(merged):
                    nxt = list(DECIMAL_CURRENCY_RE.finditer(merged[i + 1]["text"]))
                    if nxt: price_str = nxt[-1].group(1)
                if price_str:
                    if _looks_like_date_fragment(price_str, full_text): continue
                    parsed = _parse_amount(price_str)
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
        b = candidates[0]
        return b["value"], b["confidence"]
    # Fallback — bottom-half largest plausible
    all_amounts = []
    for line in merged:
        text = line["text"].strip()
        if re.search(r"\d{6,}", text): continue
        if re.search(r"[A-Z]{2,}\d{4,}", text): continue
        if len(re.findall(r"\d", text)) > 12: continue
        for m in DECIMAL_CURRENCY_RE.finditer(text):
            raw = m.group(1)
            parsed = _parse_amount(raw)
            if parsed is None or not (0.01 <= parsed <= 100_000): continue
            if _looks_like_date_fragment(raw, full_text): continue
            all_amounts.append((parsed, line["confidence"], line["y_center"]))
    if all_amounts:
        counter = Counter(round(v, 2) for v, _, _ in all_amounts)
        common, freq = counter.most_common(1)[0]
        if freq >= 2:
            conf = max(c for v, c, _ in all_amounts if round(v, 2) == common)
            return f"{common:.2f}", conf * 0.7
        n = len(merged)
        if n > 1:
            mid_y = merged[n // 2]["y_center"]
            bottom = [(v, c) for v, c, y in all_amounts if y >= mid_y]
            if bottom:
                bottom.sort(key=lambda x: -x[0])
                return f"{bottom[0][0]:.2f}", bottom[0][1] * 0.6
        all_amounts.sort(key=lambda x: -x[0])
        v, c, _ = all_amounts[0]
        return f"{v:.2f}", c * 0.45
    return None


def _find_keyword_value_with_bbox(lines: List[Dict], keywords: list,
                                    full_text: str,
                                    max_value: float = 1_000_000) -> Optional[Tuple[float, float]]:
    for kw_line in lines:
        text_lower = kw_line["text"].lower()
        kw_match = None
        for kw in keywords:
            if kw in text_lower:
                kw_match = kw; break
        if not kw_match: continue
        if "invoice" in text_lower or "exempt" in text_lower: continue
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
        kw_x_end = max(pt[0] for pt in kw_line["bbox"]) if kw_line.get("bbox") else 0
        kw_y = kw_line["y_center"]
        cands = []
        for other in lines:
            if other is kw_line: continue
            if not _vertically_overlap(kw_line, other, min_overlap=0.4): continue
            x_start = min(pt[0] for pt in other["bbox"]) if other.get("bbox") else 0
            if x_start < kw_x_end: continue
            mm = DECIMAL_CURRENCY_RE.search(other["text"])
            if not mm: continue
            ps = mm.group(1)
            if _looks_like_date_fragment(ps, full_text): continue
            pv = _parse_amount(ps)
            if pv is None or not (0.01 <= pv < max_value): continue
            cands.append((abs(other["y_center"] - kw_y), pv, other["confidence"]))
        if cands:
            cands.sort(key=lambda c: c[0])
            _, val, conf = cands[0]
            return val, conf
    return None


def extract_subtotal(lines: List[Dict]) -> Optional[Tuple[str, float]]:
    full_text = " ".join(l["text"] for l in lines)
    res = _find_keyword_value_with_bbox(
        lines, ["subtotal", "sub total", "sub-total"], full_text)
    if res:
        v, c = res
        return f"{v:.2f}", c * 0.92
    return None


def extract_tax(lines: List[Dict]) -> Optional[Tuple[str, float]]:
    full_text = " ".join(l["text"] for l in lines)
    keywords = ["service tax", "cgst", "sgst", "igst", "gst", "vat", "tax"]
    res = _find_keyword_value_with_bbox(lines, keywords, full_text, max_value=100_000)
    if res:
        v, c = res
        return f"{v:.2f}", c * 0.90
    return None


def detect_currency(lines: List[Dict]) -> Tuple[str, str]:
    full_text = " ".join(l["text"] for l in lines)
    full_lower = full_text.lower()
    if "₹" in full_text: return "₹", "INR"
    if "€" in full_text: return "€", "EUR"
    if "£" in full_text: return "£", "GBP"
    if "¥" in full_text: return "¥", "JPY"
    if re.search(r"\bRM\b|ringgit", full_text): return "RM", "MYR"
    if re.search(r"\bINR\b|rupee", full_lower): return "₹", "INR"
    if re.search(r"\bRs\.?\b", full_text): return "₹", "INR"
    if re.search(r"\bUSD\b|us\s*dollar", full_lower): return "$", "USD"
    if re.search(r"\bEUR\b|euro", full_lower): return "€", "EUR"
    if re.search(r"\bGBP\b|pound", full_lower): return "£", "GBP"
    if re.search(r"\bSGD\b|singapore\s*dollar", full_lower): return "S$", "SGD"
    if re.search(r"walmart|target|costco|cvs|walgreens", full_lower): return "$", "USD"
    if re.search(r"big\s*bazaar|reliance|dmart|tata", full_lower): return "₹", "INR"
    if "$" in full_text: return "$", "USD"
    return "$", "USD"


def extract_items(lines: List[Dict]) -> List[Dict]:
    merged = _merge_lines_into_rows(lines)
    full_text = " ".join(l["text"] for l in merged)
    items = []
    PRICE_LINE_RE = re.compile(
        r"^(.+?)\s+(?:₹|Rs\.?|RM|\$)?\s*(\d+(?:[,]\d{3})*\.\d{2}|\d+\.\d{2})\s*[A-Z]?\s*$"
    )
    for line in merged:
        text = line["text"].strip()
        text_lower = text.lower()
        if any(k in text_lower for k in ITEM_SKIP_KEYWORDS): continue
        if re.search(r"\d{6,}", text): continue
        if re.search(r"[A-Z]{2,}\d{4,}", text): continue
        m = PRICE_LINE_RE.match(text)
        if not m: continue
        name = m.group(1).strip()
        price_raw = m.group(2)
        if _looks_like_date_fragment(price_raw, full_text): continue
        price = _parse_amount(price_raw)
        if price is None or price <= 0: continue
        alpha_count = sum(1 for c in name if c.isalpha())
        if alpha_count < 3 or len(name) > 80: continue
        if _alpha_ratio(name) < 0.4: continue
        if name.count("#") >= 2 or name.count(":") >= 2: continue
        items.append({
            "name": name,
            "price": f"{price:.2f}",
            "confidence": line["confidence"] * 0.78,
        })
    return items


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CACHED PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _warm_ocr_reader():
    return _get_reader()


@st.cache_data(show_spinner=False, max_entries=100)
def _cached_extract(file_bytes: bytes, filename: str) -> dict:
    arr = np.frombuffer(file_bytes, np.uint8)
    img_orig = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_orig is None:
        return {"error": "Cannot decode image"}
    gray = to_grayscale(img_orig)
    blur_score = estimate_blur(gray)
    img = resize_for_ocr(img_orig, target_height=1200, max_height=1800)
    img = enhance_contrast(img)
    if blur_score < 200: img = denoise(img)
    img, _ = deskew(img)
    lines = run_ocr(img)
    if not lines:
        return {"error": "No text detected"}
    avg_ocr = get_average_confidence(lines)
    store_name, store_raw = extract_store_name(lines)
    date_res     = extract_date(lines)
    total_res    = extract_total(lines)
    subtotal_res = extract_subtotal(lines)
    tax_res      = extract_tax(lines)
    items_raw    = extract_items(lines)
    csym, ccode  = detect_currency(lines)
    date_v     = date_res[0]     if date_res     else None
    total_v    = total_res[0]    if total_res    else None
    subtotal_v = subtotal_res[0] if subtotal_res else None
    tax_v      = tax_res[0]      if tax_res      else None
    store_c = adjust_confidence("store_name", store_name, store_raw)
    date_c  = adjust_confidence("date", date_v, date_res[1] if date_res else 0)
    total_c = adjust_confidence("total_amount", total_v, total_res[1] if total_res else 0)
    overall = (store_c * 0.25 + date_c * 0.25 + total_c * 0.5)
    carbon = 0.0
    try:
        if total_v: carbon = float(total_v) * CARBON_KG_PER_UNIT
    except (TypeError, ValueError):
        carbon = 0.0
    return {
        "filename": filename,
        "store_name": store_name,
        "date": date_v,
        "total_amount": total_v,
        "subtotal": subtotal_v,
        "tax": tax_v,
        "currency_symbol": csym,
        "currency_code":   ccode,
        "store_conf": store_c,
        "date_conf":  date_c,
        "total_conf": total_c,
        "overall_conf": overall,
        "avg_ocr": avg_ocr,
        "carbon_kg": round(carbon, 3),
        "n_items": len(items_raw),
        "items": items_raw[:8],
    }


def process_file(file_bytes: bytes, filename: str) -> dict:
    result = _cached_extract(file_bytes, filename)
    if "error" in result:
        return result
    arr = np.frombuffer(file_bytes, np.uint8)
    img_orig = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return {**result, "_original_image": img_orig}


try:
    _warm_ocr_reader()
except Exception:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CSS — Carbon Crunch design system
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@500;600&display=swap');

:root {
    --bg-0: #050a08; --bg-1: #0a1410; --bg-2: #0f1d18;
    --line: rgba(0, 200, 150, 0.12);
    --line-strong: rgba(0, 200, 150, 0.28);
    --accent: #00c896;
    --accent-2: #4ade80;
    --accent-glow: rgba(0, 200, 150, 0.4);
    --text-0: #f0fdf4; --text-1: #d1fae5;
    --text-2: #94d4b3; --text-3: #6b8a7c; --text-4: #4b6359;
    --danger: #ef4444; --warn: #f59e0b;
}

* { box-sizing: border-box; }
.stApp {
    background:
      radial-gradient(ellipse 1100px 750px at 75% -10%, rgba(0, 200, 150, 0.12) 0%, transparent 60%),
      radial-gradient(ellipse 800px 500px at 0% 110%, rgba(0, 200, 150, 0.07) 0%, transparent 60%),
      var(--bg-0);
    background-attachment: fixed;
    font-family: 'Inter', -apple-system, sans-serif;
    color: var(--text-1);
}
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
    display: none !important; height: 0 !important;
}
#MainMenu, footer { visibility: hidden; }
.block-container { padding: 1rem 2rem 4rem !important; max-width: 1280px !important; }
.stMarkdown a[href^="#"], h1 a, h2 a, h3 a, h4 a,
[data-testid="stHeaderActionElements"], [class*="StyledLinkIcon"] {
    display: none !important; visibility: hidden !important;
}
.stMarkdown p { margin: 0 !important; }
code, pre { font-family: 'JetBrains Mono', monospace !important; }

@keyframes pulse  { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
@keyframes glow   { 0%,100% { box-shadow: 0 0 20px rgba(0,200,150,0.25); }
                    50%      { box-shadow: 0 0 32px rgba(0,200,150,0.45); } }
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
@keyframes growBar{ from { width: 0%; } }
@keyframes float  { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
@keyframes co2Glow {
    0%,100% { box-shadow: 0 0 28px rgba(0,200,150,0.25), inset 0 0 0 1px rgba(0,200,150,0.15); }
    50%     { box-shadow: 0 0 48px rgba(0,200,150,0.45), inset 0 0 0 1px rgba(0,200,150,0.3); }
}
@keyframes shimmer{ 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }

.nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.85rem 1.3rem; background: rgba(8, 18, 14, 0.7);
    backdrop-filter: blur(24px); border: 1px solid var(--line);
    border-radius: 14px; margin-bottom: 3rem; animation: fadeIn 0.5s ease;
}
.brand { display: flex; align-items: center; gap: 0.75rem; }
.brand-mark {
    width: 38px; height: 38px; border-radius: 11px;
    background: linear-gradient(135deg, var(--accent), #00a884);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 16px var(--accent-glow);
}
.brand-mark svg { width: 20px; height: 20px; }
.brand-text { line-height: 1.1; }
.brand-text .name { font-size: 1rem; font-weight: 700; color: var(--text-0); letter-spacing: -0.01em; }
.brand-text .subtitle { font-size: 0.72rem; color: var(--accent-2); font-weight: 500; opacity: 0.85; margin-top: 2px; }
.nav-pill {
    background: rgba(0, 200, 150, 0.1); border: 1px solid var(--line-strong);
    color: var(--accent-2); padding: 0.32rem 0.85rem; border-radius: 999px;
    font-size: 0.66rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
}
.nav-pill.muted { background: rgba(255,255,255,0.04); border-color: rgba(255,255,255,0.08); color: var(--text-3); }
.status-dot {
    width: 7px; height: 7px; background: var(--accent); border-radius: 50%;
    box-shadow: 0 0 8px var(--accent); animation: pulse 2s ease-in-out infinite;
    display: inline-block; margin-right: 0.4rem; vertical-align: middle;
}
.nav-meta { display: flex; gap: 0.5rem; align-items: center; }

.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(0, 200, 150, 0.1); border: 1px solid var(--line-strong);
    color: var(--accent-2); padding: 0.4rem 1rem; border-radius: 999px;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.15em;
    text-transform: uppercase; margin-bottom: 1.6rem; animation: fadeIn 0.6s ease;
}
.hero-h1 {
    font-size: 3.2rem; font-weight: 800; line-height: 1.08;
    letter-spacing: -0.04em; color: var(--text-0); margin: 0 0 1.4rem 0;
    animation: fadeIn 0.7s ease;
}
.hero-h1 .accent {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub {
    color: var(--text-2); font-size: 1.05rem; line-height: 1.65;
    margin: 0 0 2rem 0; max-width: 480px; opacity: 0.9; animation: fadeIn 0.8s ease;
}
.feat-tags { display: flex; gap: 0.6rem; flex-wrap: wrap; animation: fadeIn 0.9s ease; }
.feat-tag {
    background: rgba(8, 18, 14, 0.6); border: 1px solid var(--line);
    color: var(--text-1); padding: 0.5rem 0.95rem; border-radius: 10px;
    font-size: 0.78rem; font-weight: 500;
    display: inline-flex; align-items: center; gap: 0.45rem; transition: all 0.2s;
}
.feat-tag:hover { border-color: var(--line-strong); background: rgba(0, 200, 150, 0.06); transform: translateY(-1px); }
.feat-tag .label { color: var(--accent-2); font-weight: 600; }
.feat-tag svg { width: 14px; height: 14px; color: var(--accent-2); }

.hero-art {
    position: relative; height: 380px; width: 100%;
    display: flex; align-items: center; justify-content: center;
    animation: fadeIn 1s ease;
}

.section-eyebrow { color: var(--accent-2); font-size: 0.7rem; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase; text-align: center; margin-bottom: 0.8rem; }
.section-title   { color: var(--text-0); font-size: 1.95rem; font-weight: 700; letter-spacing: -0.02em; text-align: center; margin: 0 0 0.7rem 0; }
.section-sub     { color: var(--text-2); font-size: 0.95rem; text-align: center; margin: 0 auto 2.2rem auto; max-width: 540px; opacity: 0.85; line-height: 1.6; }

[data-testid="stFileUploaderDropzone"] {
    background: rgba(8, 18, 14, 0.55) !important;
    border: 2px dashed var(--line-strong) !important;
    border-radius: 18px !important; padding: 3rem 2rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important; background: rgba(0, 200, 150, 0.05) !important;
    box-shadow: 0 0 40px rgba(0, 200, 150, 0.18); transform: scale(1.005);
}
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] div { color: var(--text-2) !important; }
.format-badges { display: flex; justify-content: center; gap: 0.5rem; margin-top: 1rem; flex-wrap: wrap; }
.format-badge {
    background: rgba(0, 200, 150, 0.06); border: 1px solid var(--line);
    color: var(--accent-2); padding: 0.3rem 0.75rem; border-radius: 6px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.05em;
}

.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, var(--accent), #00a884) !important;
    color: #052218 !important; border: none !important; border-radius: 11px !important;
    padding: 0.78rem 1.5rem !important; font-weight: 700 !important;
    font-size: 0.9rem !important;
    box-shadow: 0 4px 16px rgba(0, 200, 150, 0.3) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 10px 28px rgba(0, 200, 150, 0.45) !important;
}

.result-card {
    background: linear-gradient(135deg, rgba(0, 200, 150, 0.04), rgba(8, 18, 14, 0.65));
    border: 1px solid var(--line-strong); border-radius: 16px;
    padding: 1.8rem; backdrop-filter: blur(24px);
    position: relative; overflow: hidden; animation: fadeIn 0.5s ease;
}
.result-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-glow), transparent);
}
.result-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid var(--line);
}
.result-title {
    color: var(--text-0); font-size: 1.1rem; font-weight: 700; letter-spacing: -0.01em;
}
.result-title .icon { display: inline-flex; vertical-align: middle; margin-right: 0.5rem; color: var(--accent-2); }
.confidence-pill {
    display: inline-flex; align-items: center; gap: 0.45rem;
    padding: 0.35rem 0.85rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700;
    border: 1px solid var(--line-strong); background: rgba(0, 200, 150, 0.1);
    color: var(--accent-2); font-family: 'JetBrains Mono', monospace;
}
.confidence-pill .dot { width: 6px; height: 6px; background: var(--accent); border-radius: 50%; box-shadow: 0 0 8px var(--accent); animation: pulse 2s ease-in-out infinite; }

.co2-hero {
    background:
      radial-gradient(circle at 20% 30%, rgba(0, 200, 150, 0.2) 0%, transparent 60%),
      linear-gradient(135deg, rgba(0, 200, 150, 0.18) 0%, rgba(74, 222, 128, 0.06) 100%);
    border: 1px solid rgba(0, 200, 150, 0.45); border-radius: 16px;
    padding: 1.8rem 1.5rem; margin-bottom: 1.4rem;
    position: relative; overflow: hidden; animation: co2Glow 3s ease-in-out infinite;
}
.co2-hero::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
}
.co2-hero::after {
    content: ''; position: absolute; top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(0,200,150,0.18), transparent 70%);
    pointer-events: none; animation: float 5s ease-in-out infinite;
}
.co2-hero-row { display: flex; align-items: center; gap: 1.2rem; position: relative; z-index: 1; }
.co2-hero-icon {
    width: 64px; height: 64px; flex-shrink: 0;
    background: linear-gradient(135deg, rgba(0, 200, 150, 0.25), rgba(0, 200, 150, 0.1));
    border: 1px solid rgba(0, 200, 150, 0.4); border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 8px 24px rgba(0, 200, 150, 0.25);
}
.co2-hero-icon svg { width: 32px; height: 32px; color: var(--accent-2); }
.co2-hero-content { flex: 1; }
.co2-hero-label {
    color: var(--accent-2); font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 0.4rem;
    display: flex; align-items: center; justify-content: space-between;
}
.co2-hero-label .factor-tag {
    background: rgba(0, 200, 150, 0.1); border: 1px solid rgba(0, 200, 150, 0.25);
    color: var(--accent-2); padding: 0.15rem 0.55rem; border-radius: 6px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; font-weight: 600;
    letter-spacing: 0.04em; text-transform: none;
}
.co2-hero-value {
    color: var(--text-0); font-size: 3rem; font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.04em; line-height: 1;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-2) 60%, var(--accent) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.co2-hero-value .unit {
    font-size: 1.1rem; opacity: 0.7;
    color: var(--accent-2); -webkit-text-fill-color: var(--accent-2);
    margin-left: 0.45rem; font-weight: 500;
}
.co2-hero-context {
    margin-top: 0.9rem; color: var(--text-3); font-size: 0.78rem; line-height: 1.5;
    position: relative; z-index: 1;
}
.co2-hero-context strong { color: var(--text-1); font-weight: 600; font-family: 'JetBrains Mono', monospace; }

.money-input {
    background: rgba(8, 18, 14, 0.6); border: 1px solid var(--line);
    border-radius: 12px; padding: 1.1rem 1.3rem; margin-bottom: 1.4rem;
}
.money-input-label {
    color: var(--text-3); font-size: 0.66rem; font-weight: 700;
    letter-spacing: 0.16em; text-transform: uppercase; margin-bottom: 0.7rem;
    display: flex; align-items: center; justify-content: space-between;
}
.money-input-label .currency-tag {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    color: var(--text-2); padding: 0.15rem 0.55rem; border-radius: 6px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; font-weight: 600; letter-spacing: 0.04em;
}
.money-input-value {
    color: var(--text-1); font-size: 1.6rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace; letter-spacing: -0.02em; line-height: 1;
}
.money-input-value .symbol { color: var(--text-3); font-size: 1.1rem; margin-right: 0.25rem; font-weight: 500; }
.money-input-meta { margin-top: 0.7rem; color: var(--text-3); font-size: 0.75rem; display: flex; gap: 1.2rem; flex-wrap: wrap; }
.money-input-meta .key { color: var(--text-3); font-weight: 500; }
.money-input-meta .val { color: var(--text-1); font-weight: 600; font-family: 'JetBrains Mono', monospace; margin-left: 0.3rem; }

.breakdown {
    background: rgba(8, 18, 14, 0.5); border: 1px solid var(--line);
    border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 1.4rem;
}
.breakdown-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.5rem 0; border-bottom: 1px solid rgba(0, 200, 150, 0.05);
}
.breakdown-row:last-child {
    border-bottom: none; padding-top: 0.7rem; margin-top: 0.2rem;
    border-top: 1px solid var(--line-strong);
}
.breakdown-label { color: var(--text-3); font-size: 0.78rem; font-weight: 500; }
.breakdown-value { color: var(--text-1); font-size: 0.9rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.breakdown-row.is-total .breakdown-label { color: var(--accent-2); font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.7rem; }
.breakdown-row.is-total .breakdown-value { color: var(--text-0); font-size: 1.05rem; font-weight: 800; }

.field { padding: 0.95rem 0; border-bottom: 1px solid rgba(0, 200, 150, 0.06); animation: fadeIn 0.4s ease; }
.field:last-of-type { border-bottom: none; }
.field-row { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin-bottom: 0.5rem; }
.field-label { color: var(--text-3); font-size: 0.7rem; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase; }
.field-value { color: var(--text-0); font-size: 1rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 260px; }
.conf-bar-container { display: flex; align-items: center; gap: 0.7rem; margin-top: 0.4rem; }
.conf-bar { flex: 1; height: 5px; background: rgba(255,255,255,0.05); border-radius: 999px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 999px; animation: growBar 0.8s ease-out; transition: width 0.3s; }
.conf-bar-fill.high { background: linear-gradient(90deg, var(--accent), var(--accent-2)); }
.conf-bar-fill.mid  { background: linear-gradient(90deg, var(--warn), #fbbf24); }
.conf-bar-fill.low  { background: linear-gradient(90deg, var(--danger), #f87171); }
.conf-pct { color: var(--text-1); font-size: 0.75rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; min-width: 40px; text-align: right; }

.preview-wrap {
    background: rgba(8, 18, 14, 0.55); border: 1px solid var(--line);
    border-radius: 16px; padding: 1.2rem; transition: all 0.25s; animation: fadeIn 0.5s ease;
}
.preview-wrap:hover { border-color: var(--line-strong); transform: translateY(-1px); }
.preview-label { color: var(--text-3); font-size: 0.66rem; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase; margin-bottom: 0.9rem; display: flex; align-items: center; gap: 0.4rem; }
.preview-label .file { color: var(--text-1); text-transform: none; letter-spacing: 0.01em; }
[data-testid="stImage"] img { border-radius: 12px; border: 1px solid var(--line); }

.step {
    background: rgba(8, 18, 14, 0.55); border: 1px solid var(--line);
    border-radius: 16px; padding: 1.8rem 1.4rem; text-align: center; height: 100%;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); animation: fadeIn 0.6s ease;
}
.step:hover {
    transform: translateY(-4px); border-color: var(--line-strong);
    box-shadow: 0 16px 40px rgba(0, 200, 150, 0.12); background: rgba(0, 200, 150, 0.03);
}
.step-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 40px; height: 40px;
    background: linear-gradient(135deg, var(--accent), #00a884);
    color: #052218; font-weight: 800; font-size: 1rem;
    border-radius: 11px; margin: 0 auto 1rem auto;
    box-shadow: 0 4px 16px var(--accent-glow);
}
.step-title { color: var(--text-0); font-size: 1.05rem; font-weight: 700; margin: 0 0 0.5rem 0; letter-spacing: -0.01em; }
.step-text  { color: var(--text-2); font-size: 0.88rem; line-height: 1.6; opacity: 0.88; }

.empty { text-align: center; padding: 4rem 2rem; animation: fadeIn 0.5s ease; }
.empty-icon { width: 60px; height: 60px; margin: 0 auto 1.1rem; background: rgba(0, 200, 150, 0.08); border: 1px solid var(--line-strong); border-radius: 15px; display: flex; align-items: center; justify-content: center; }
.empty-icon svg { width: 28px; height: 28px; color: var(--accent-2); }
.empty-title { color: var(--text-1); font-size: 1.05rem; font-weight: 500; margin-bottom: 0.4rem; }
.empty-sub   { color: var(--text-3); font-size: 0.82rem; }

.app-footer { text-align: center; color: var(--text-4); font-size: 0.78rem; padding: 2.5rem 0 0.5rem; border-top: 1px solid rgba(0, 200, 150, 0.06); margin-top: 4.5rem; }
.app-footer strong { color: var(--accent-2); font-weight: 600; }

.stSpinner > div > div { border-top-color: var(--accent) !important; }
.stProgress > div > div > div > div { background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important; background-size: 200% 100% !important; animation: shimmer 2s linear infinite !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def conf_class(s: float) -> str:
    return "high" if s >= 0.7 else ("mid" if s >= 0.5 else "low")


def render_field(label: str, value: str, conf: Optional[float] = None) -> str:
    val = value if value not in (None, "") else "—"
    if conf is None:
        return (f'<div class="field"><div class="field-row">'
                f'<span class="field-label">{label}</span>'
                f'<span class="field-value">{val}</span></div></div>')
    pct = int(conf * 100); cls = conf_class(conf)
    return (f'<div class="field"><div class="field-row">'
            f'<span class="field-label">{label}</span>'
            f'<span class="field-value">{val}</span></div>'
            f'<div class="conf-bar-container">'
            f'<div class="conf-bar"><div class="conf-bar-fill {cls}" style="width:{pct}%"></div></div>'
            f'<span class="conf-pct">{pct}%</span></div></div>')


# ═══════════════════════════════════════════════════════════════════════════════
# 8. NAVBAR
# ═══════════════════════════════════════════════════════════════════════════════
LOGO_SVG = """
<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M12 2C8 6 4 9 4 14C4 18.4 7.6 22 12 22C16.4 22 20 18.4 20 14C20 9 16 6 12 2Z"
        fill="#052218" stroke="#ecfdf5" stroke-width="1.6" stroke-linejoin="round"/>
  <path d="M12 11C10 13 8 15 8 17" stroke="#4ade80" stroke-width="1.6" stroke-linecap="round"/>
</svg>
"""

st.markdown(f"""
<div class="nav">
  <div class="brand">
    <div class="brand-mark">{LOGO_SVG}</div>
    <div class="brand-text">
      <div class="name">Carbon Crunch</div>
      <div class="subtitle">Receipt Intelligence</div>
    </div>
  </div>
  <div class="nav-meta">
    <span class="nav-pill"><span class="status-dot"></span>Pipeline Live</span>
    <span class="nav-pill muted">v1.0</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. HERO
# ═══════════════════════════════════════════════════════════════════════════════
hero_l, hero_r = st.columns([1.05, 1], gap="large")

with hero_l:
    st.markdown("""
    <div class="hero-eyebrow">◆ AI · OCR · CARBON ANALYTICS</div>
    <div class="hero-h1">
      Turn receipts into <span class="accent">carbon insights</span> instantly.
    </div>
    <div class="hero-sub">
      Upload receipts and get structured data with carbon estimates in seconds.
      Confidence-aware extraction, audit-ready outputs.
    </div>
    <div class="feat-tags">
      <div class="feat-tag">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
          <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
          <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
        </svg>
        <span class="label">OCR</span> EasyOCR engine
      </div>
      <div class="feat-tag">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
          <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
          <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
        </svg>
        <span class="label">Privacy</span> on-device
      </div>
      <div class="feat-tag">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
          <polyline points="20 6 9 17 4 12"/>
        </svg>
        <span class="label">Audit</span> confidence-tagged
      </div>
    </div>
    """, unsafe_allow_html=True)

with hero_r:
    st.markdown("""
    <div class="hero-art">
      <svg viewBox="0 0 400 380" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:100%">
        <defs>
          <linearGradient id="recBg" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#0a1f17"/>
            <stop offset="100%" stop-color="#081813"/>
          </linearGradient>
        </defs>
        <g transform="translate(50 30) rotate(-7 110 160)">
          <path d="M0 0 L220 0 L220 290 L210 300 L195 290 L180 300 L165 290 L150 300 L135 290 L120 300 L105 290 L90 300 L75 290 L60 300 L45 290 L30 300 L15 290 L0 300 Z" fill="url(%23recBg)" stroke="rgba(0,200,150,0.45)" stroke-width="1.5"/>
          <line x1="20" y1="38"  x2="200" y2="38"  stroke="#4ade80" stroke-width="2.5"/>
          <line x1="20" y1="58"  x2="160" y2="58"  stroke="rgba(255,255,255,0.45)" stroke-width="1.5"/>
          <line x1="20" y1="88"  x2="100" y2="88"  stroke="rgba(255,255,255,0.22)" stroke-width="1.4"/>
          <line x1="120" y1="88" x2="200" y2="88"  stroke="rgba(255,255,255,0.22)" stroke-width="1.4"/>
          <line x1="20" y1="113" x2="100" y2="113" stroke="rgba(255,255,255,0.22)" stroke-width="1.4"/>
          <line x1="120" y1="113" x2="200" y2="113" stroke="rgba(255,255,255,0.22)" stroke-width="1.4"/>
          <line x1="20" y1="138" x2="100" y2="138" stroke="rgba(255,255,255,0.22)" stroke-width="1.4"/>
          <line x1="120" y1="138" x2="200" y2="138" stroke="rgba(255,255,255,0.22)" stroke-width="1.4"/>
          <line x1="20" y1="180" x2="200" y2="180" stroke="rgba(0,200,150,0.4)" stroke-width="1.2" stroke-dasharray="3"/>
          <line x1="20" y1="208" x2="100" y2="208" stroke="rgba(74,222,128,0.7)" stroke-width="2.5"/>
          <line x1="120" y1="208" x2="200" y2="208" stroke="rgba(74,222,128,0.95)" stroke-width="3"/>
        </g>
        <g transform="translate(225 50)" style="animation:float 4s ease-in-out infinite">
          <rect x="0" y="0" width="150" height="74" rx="12" fill="rgba(8,18,14,0.95)" stroke="rgba(0,200,150,0.55)" stroke-width="1.3"/>
          <circle cx="20" cy="24" r="7" fill="#4ade80"/>
          <rect x="34" y="18" width="80" height="5" rx="2" fill="rgba(255,255,255,0.55)"/>
          <rect x="34" y="28" width="50" height="4" rx="2" fill="rgba(255,255,255,0.28)"/>
          <line x1="14" y1="46" x2="136" y2="46" stroke="rgba(0,200,150,0.18)"/>
          <text x="14" y="63" font-family="monospace" font-size="11" fill="#4ade80" font-weight="700">986.50</text>
          <text x="130" y="63" font-family="monospace" font-size="9" fill="rgba(74,222,128,0.75)" text-anchor="end">94%</text>
        </g>
        <g transform="translate(245 200)" style="animation:float 4s ease-in-out infinite 0.5s">
          <rect x="0" y="0" width="155" height="86" rx="12" fill="rgba(8,18,14,0.95)" stroke="rgba(0,200,150,0.55)" stroke-width="1.3"/>
          <g transform="translate(14 18)">
            <path d="M14 0C9 5 4 9 4 16C4 22 9 26 14 26C19 26 24 22 24 16C24 9 19 5 14 0Z" fill="rgba(0,200,150,0.22)" stroke="#4ade80" stroke-width="1.5"/>
            <path d="M14 12C12 14 10 16 10 18" stroke="#4ade80" stroke-width="1.5" stroke-linecap="round"/>
          </g>
          <text x="50" y="22" font-family="Inter" font-size="9" fill="rgba(255,255,255,0.55)" font-weight="700" letter-spacing="2">CO2 EST.</text>
          <text x="50" y="46" font-family="monospace" font-size="20" fill="#4ade80" font-weight="700">40.4 kg</text>
          <rect x="14" y="62" width="125" height="6" rx="3" fill="rgba(255,255,255,0.07)"/>
          <rect x="14" y="62" width="80"  height="6" rx="3" fill="#00c896" opacity="0.85"/>
        </g>
        <circle cx="200" cy="180" r="3" fill="#00c896" opacity="0.7"/>
        <circle cx="218" cy="170" r="2" fill="#00c896" opacity="0.5"/>
        <circle cx="238" cy="155" r="2.5" fill="#00c896" opacity="0.6"/>
      </svg>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div style="margin-top:3.5rem"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-eyebrow">◇ Step 1 — Upload</div>
<div class="section-title">Drop receipts or browse</div>
<div class="section-sub">
  Process one or many at once. Images stay on the server —
  extraction takes about 5 seconds per receipt.
</div>
""", unsafe_allow_html=True)

up_l, up_c, up_r = st.columns([1, 6, 1])
with up_c:
    files = st.file_uploader(
        "upload",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    st.markdown("""
    <div class="format-badges">
      <span class="format-badge">PNG</span>
      <span class="format-badge">JPG</span>
      <span class="format-badge">JPEG</span>
      <span class="format-badge">BMP</span>
      <span class="format-badge">TIFF</span>
      <span class="format-badge">WEBP</span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. PROCESSING + RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
if files:
    sig = [f.name + str(f.size) for f in files]
    if st.session_state.get("file_signature") != sig:
        with st.spinner(f"◆ Running AI OCR pipeline on {len(files)} receipt(s)…"):
            results = []
            for f in files:
                results.append(process_file(f.read(), f.name))
                time.sleep(0.05)
            st.session_state["results"] = results
            st.session_state["file_signature"] = sig
    results = st.session_state["results"]

    st.markdown('<div style="margin-top:3rem"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-eyebrow">◇ Step 2 — Insights</div>
    <div class="section-title">Extracted data + carbon estimates</div>
    <div class="section-sub">
      Every field comes with a calibrated confidence score.
      Carbon estimate uses an industry-average emission coefficient.
    </div>
    """, unsafe_allow_html=True)

    for idx, r in enumerate(results):
        if "error" in r:
            st.error(f"❌ {r.get('filename', 'file')}: {r['error']}")
            continue

        st.markdown('<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
        col_img, col_data = st.columns([1, 1.15], gap="large")

        with col_img:
            st.markdown(
                f'<div class="preview-wrap">'
                f'<div class="preview-label">'
                f'◆ Receipt <span class="file">· {r["filename"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.image(cv2.cvtColor(r["_original_image"], cv2.COLOR_BGR2RGB),
                     use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_data:
            opct = int(r["overall_conf"] * 100)
            sym  = r["currency_symbol"]
            code = r["currency_code"]
            total_str = r["total_amount"] or "—"
            subtotal_str = r["subtotal"] or "—"
            tax_str = r["tax"] or "—"

            breakdown_rows = ""
            if r["subtotal"]:
                breakdown_rows += (
                    '<div class="breakdown-row">'
                    '<span class="breakdown-label">Subtotal</span>'
                    f'<span class="breakdown-value">{sym} {subtotal_str}</span>'
                    '</div>'
                )
            if r["tax"]:
                breakdown_rows += (
                    '<div class="breakdown-row">'
                    '<span class="breakdown-label">Tax / GST</span>'
                    f'<span class="breakdown-value">{sym} {tax_str}</span>'
                    '</div>'
                )
            breakdown_rows += (
                '<div class="breakdown-row is-total">'
                '<span class="breakdown-label">Total</span>'
                f'<span class="breakdown-value">{sym} {total_str}</span>'
                '</div>'
            )
            breakdown_html = ""
            if r["subtotal"] or r["tax"]:
                breakdown_html = f'<div class="breakdown">{breakdown_rows}</div>'

            carbon = r["carbon_kg"]
            if carbon > 0:
                km_equiv = carbon / 0.18
                tree_days = carbon / 0.06
                ctx = (f'Equivalent to <strong>{km_equiv:.1f} km</strong> of car driving '
                       f'or <strong>{tree_days:.1f} days</strong> of CO₂ absorbed by one tree.')
            else:
                ctx = "Upload a receipt with a detected total to see emission estimate."

            html = (
                '<div class="result-card">'
                  '<div class="result-header">'
                    '<div class="result-title">'
                      '<svg class="icon" width="18" height="18" viewBox="0 0 24 24" fill="none" '
                      'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                      '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
                      '<polyline points="14 2 14 8 20 8"/>'
                      '<line x1="16" y1="13" x2="8" y2="13"/>'
                      '<line x1="16" y1="17" x2="8" y2="17"/>'
                      '</svg>Receipt Insights'
                    '</div>'
                    f'<div class="confidence-pill"><span class="dot"></span>'
                    f'Confidence {opct}%</div>'
                  '</div>'
                  '<div class="co2-hero">'
                    '<div class="co2-hero-row">'
                      '<div class="co2-hero-icon">'
                        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
                        '<path d="M12 2C8 6 4 9 4 14C4 18.4 7.6 22 12 22C16.4 22 20 18.4 20 14C20 9 16 6 12 2Z"/>'
                        '<path d="M12 11C10 13 8 15 8 17"/>'
                        '</svg>'
                      '</div>'
                      '<div class="co2-hero-content">'
                        '<div class="co2-hero-label">'
                          '<span>Estimated Carbon Footprint</span>'
                          '<span class="factor-tag">factor: 0.041 kg/$</span>'
                        '</div>'
                        f'<div class="co2-hero-value">{carbon:.2f}<span class="unit">kg CO₂e</span></div>'
                      '</div>'
                    '</div>'
                    f'<div class="co2-hero-context">{ctx}</div>'
                  '</div>'
                  '<div class="money-input">'
                    '<div class="money-input-label">'
                      '<span>◆ Money Spent (input)</span>'
                      f'<span class="currency-tag">{code}</span>'
                    '</div>'
                    f'<div class="money-input-value"><span class="symbol">{sym}</span>{total_str}</div>'
                    '<div class="money-input-meta">'
                      f'<span><span class="key">Store</span>'
                      f'<span class="val">{r["store_name"]}</span></span>'
                      f'<span><span class="key">Date</span>'
                      f'<span class="val">{r["date"] or "—"}</span></span>'
                      f'<span><span class="key">Items</span>'
                      f'<span class="val">{r["n_items"]}</span></span>'
                    '</div>'
                  '</div>'
                  + breakdown_html
                  +
                  '<div style="margin-bottom:0.5rem;color:var(--text-3);'
                  'font-size:0.66rem;font-weight:700;letter-spacing:0.16em;'
                  'text-transform:uppercase">◆ Field-level confidence</div>'
                  + render_field("Store", r["store_name"], r["store_conf"])
                  + render_field("Date", r["date"] or "Not detected", r["date_conf"])
                  + render_field("Total", f'{sym} {total_str}', r["total_conf"])
                  +
                '</div>'
            )
            st.markdown(html, unsafe_allow_html=True)

            payload = {
                "file": r["filename"],
                "currency": {"symbol": sym, "code": code},
                "store_name":   {"value": r["store_name"], "confidence": round(r["store_conf"], 3)},
                "date":         {"value": r["date"], "confidence": round(r["date_conf"], 3)},
                "subtotal":     {"value": r["subtotal"]},
                "tax":          {"value": r["tax"]},
                "total_amount": {"value": r["total_amount"], "confidence": round(r["total_conf"], 3)},
                "items_count":  r["n_items"],
                "carbon_estimate_kg_co2e": r["carbon_kg"],
                "overall_confidence": round(r["overall_conf"], 3),
            }
            st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
            st.download_button(
                "↓ Download JSON",
                data=json.dumps(payload, indent=2, ensure_ascii=False),
                file_name=f"{Path(r['filename']).stem}.json",
                mime="application/json",
                key=f"dl_{idx}",
                use_container_width=True,
            )
else:
    st.markdown("""
    <div class="empty">
      <div class="empty-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
        </svg>
      </div>
      <div class="empty-title">No receipts uploaded yet</div>
      <div class="empty-sub">Drop one above to see extracted insights here</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. HOW IT WORKS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div style="margin-top:5rem"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-eyebrow">◇ The Workflow</div>
<div class="section-title">How it works</div>
<div class="section-sub">
  Three steps from a paper receipt to an emissions-ready data row.
</div>
""", unsafe_allow_html=True)

s1, s2, s3 = st.columns(3, gap="large")
with s1:
    st.markdown("""
    <div class="step">
      <div class="step-num">1</div>
      <div class="step-title">Upload</div>
      <div class="step-text">
        Drop a single receipt or a batch. Supports JPG, PNG, BMP, TIFF, WebP.
      </div>
    </div>
    """, unsafe_allow_html=True)
with s2:
    st.markdown("""
    <div class="step">
      <div class="step-num">2</div>
      <div class="step-title">Extract</div>
      <div class="step-text">
        OpenCV preprocessing fixes noise, lighting, and skew. EasyOCR reads the
        text with per-line confidence. Regex + heuristics pull out the fields.
      </div>
    </div>
    """, unsafe_allow_html=True)
with s3:
    st.markdown("""
    <div class="step">
      <div class="step-num">3</div>
      <div class="step-title">Analyze</div>
      <div class="step-text">
        Each field gets a calibrated trust score, low-confidence items are
        flagged for review, and a CO₂ estimate is computed against industry
        emission factors.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 13. FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-footer">
  Built for <strong>Carbon Crunch</strong> · AI OCR Pipeline · April 2026
  <br/>
  <span style='font-size:0.7rem;opacity:0.7'>
    Crafted by Tirupathamma Guntru · EasyOCR · OpenCV · Streamlit
  </span>
</div>
""", unsafe_allow_html=True)
