"""
Carbon Crunch — AI Receipt Intelligence (ULTRA-STABLE SELF-CONTAINED EDITION)
===========================================================================
Production-quality Streamlit app. This version is completely self-contained
to resolve import failures on Streamlit Cloud.
"""
from __future__ import annotations

import json
import time
import os
import sys
import re
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CORE LOGIC (MERGED FROM INTERNAL MODULES)
# ═══════════════════════════════════════════════════════════════════════════════

# --- CONFIDENCE HELPERS ---
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

def validate_date(date_str: Optional[str]) -> float:
    if not date_str: return 0.0
    s = date_str.strip()
    for fmt in VALID_DATE_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            if 1990 <= dt.year <= 2099: return 1.0
            return 0.5
        except ValueError: continue
    if re.match(r"^\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}$", s): return 0.6
    if re.search(r"\d{1,2}[/\-.\s]\d{1,2}[/\-.\s]\d{2,4}", s): return 0.4
    return 0.2

def validate_currency(amount_str: Optional[str]) -> float:
    if not amount_str: return 0.0
    s = str(amount_str).strip().replace(",", ".")
    try:
        val = float(s)
    except ValueError: return 0.0
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
    if field == "date": factor = validate_date(value)
    elif field in ("total_amount", "price"): factor = validate_currency(value)
    elif field == "store_name": factor = validate_store_name(value)
    else: factor = 1.0
    return round(min(max(raw_conf * factor, 0.0), 1.0), 3)

# --- PREPROCESSING ---
def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3: return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        return cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return cv2.fastNlMeansDenoisingColored(image, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

def deskew(image: np.ndarray) -> Tuple[np.ndarray, float]:
    gray = to_grayscale(image)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 50: return image, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle = -(90 + angle)
    else: angle = -angle
    if abs(angle) < 0.5: return image, 0.0
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), float(angle)

def resize_for_ocr(image: np.ndarray, target_height: int = 1500) -> np.ndarray:
    h, w = image.shape[:2]
    if h < target_height:
        scale = target_height / h
        return cv2.resize(image, (int(w * scale), target_height), interpolation=cv2.INTER_CUBIC)
    return image

# --- OCR ENGINE ---
_reader = None
def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _reader

def run_ocr(image: np.ndarray) -> List[Dict]:
    reader = _get_reader()
    raw = reader.readtext(image, detail=1, paragraph=False, batch_size=8)
    lines = []
    for entry in raw:
        if len(entry) != 3: continue
        bbox, text, conf = entry
        if not text or not text.strip(): continue
        lines.append({
            "text": text.strip(),
            "confidence": float(conf),
            "bbox": [[float(p[0]), float(p[1])] for p in bbox],
            "y_center": float(np.mean([pt[1] for pt in bbox])),
        })
    return sorted(lines, key=lambda l: l["y_center"])

# --- EXTRACTION LOGIC ---
DECIMAL_CURRENCY_RE = re.compile(r"(?:(?:₹|Rs\.?|RM|INR|\$|USD|EUR|£|GBP)\s*)?(\d{1,3}(?:[,]\d{3})*\.\d{2}|\d+\.\d{2})")
TOTAL_KEYWORDS_PRIORITIZED = [("total purchase", 1), ("grand total", 100), ("total", 90), ("bal", 85), ("amount due", 80), ("net amount", 75)]
ITEM_SKIP_KEYWORDS = ["total", "subtotal", "tax", "gst", "vat", "cash", "change", "tender", "payment", "paid", "thank", "visit"]
STORE_NAME_NOISE = ["save today", "special offer", "survey", "win $", "see back"]

def _parse_amount(s: str) -> Optional[float]:
    s = str(s).strip().replace(" ", "").replace(",", "")
    try: return float(s)
    except: return None

def _vertically_overlap(a: Dict, b: Dict, min_overlap: float = 0.4) -> bool:
    a_ys = [pt[1] for pt in a["bbox"]]; b_ys = [pt[1] for pt in b["bbox"]]
    a_t, a_b = min(a_ys), max(a_ys); b_t, b_b = min(b_ys), max(b_ys)
    inter = max(0.0, min(a_b, b_b) - max(a_t, b_t))
    smaller_h = min(a_b - a_t, b_b - b_t)
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
                    rows[-1].append(line); attached = True; break
        if not attached: rows.append([line])
    merged = []
    for row in rows:
        row_sorted = sorted(row, key=lambda l: l["bbox"][0][0])
        merged.append({
            "text": " ".join(l["text"] for l in row_sorted),
            "confidence": float(np.mean([l["confidence"] for l in row_sorted])),
            "y_center": float(np.mean([l["y_center"] for l in row_sorted]))
        })
    return merged

def extract_store_name(lines: List[Dict]) -> Tuple[str, float]:
    for idx, line in enumerate(lines[:10]):
        text = line["text"].strip(); text_l = text.lower()
        if len(text) < 3 or any(n in text_l for n in STORE_NAME_NOISE): continue
        if re.search(r"walmart|spar|target|tesco", text_l):
            if "wal" in text_l and "mart" in text_l: return "Walmart", 1.0
            if "spar" in text_l: return "SPAR", 1.0
        if text.isupper() and 3 <= len(text) <= 30: return text, line["confidence"] * 0.95
    return lines[0]["text"] if lines else "Unknown", 0.5

def extract_date(lines: List[Dict]) -> Optional[Tuple[str, float]]:
    patterns = [r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})\b", r"\b(\d{1,2}\s?[/\-.]\s?\d{1,2}\s?[/\-.]\s?\d{2,4})\b"]
    for line in lines:
        for p in patterns:
            m = re.search(p, line["text"])
            if m and validate_date(m.group(1)) > 0.4: return m.group(1), line["confidence"] * 0.95
    return None

def extract_total(lines: List[Dict]) -> Optional[Tuple[str, float]]:
    for i, line in enumerate(lines):
        txt = line["text"].lower()
        for kw, prio in TOTAL_KEYWORDS_PRIORITIZED:
            if kw in txt:
                for j in range(i, min(i+3, len(lines))):
                    m = DECIMAL_CURRENCY_RE.search(lines[j]["text"].replace(",", ""))
                    if m: return m.group(1), lines[j]["confidence"] * 0.9
    return None

def extract_items(lines: List[Dict]) -> List[Dict]:
    rows = _merge_lines_into_rows(lines); items = []
    for row in rows:
        txt = row["text"]
        if any(k in txt.lower() for k in ITEM_SKIP_KEYWORDS): continue
        m = list(DECIMAL_CURRENCY_RE.finditer(txt))
        if m:
            last = m[-1]; p = _parse_amount(last.group(1))
            if p: items.append({"description": txt[:last.start()].strip(), "price": f"{p:.2f}", "quantity": 1, "confidence": row["confidence"] * 0.8})
    return items

def calculate_sum_of_items(items: List[Dict]) -> float:
    return sum(float(i["price"]) for i in items)

def generate_summary(results: List[Dict]) -> Dict:
    total = sum(float(r.get("total_amount", {}).get("value", 0)) for r in results if "error" not in r)
    return {"total_spend": round(total, 2), "total_transactions": len(results)}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. STREAMLIT APP LOGIC
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Carbon Crunch · AI Receipt Intelligence", page_icon="🌿", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stMetric { background: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #2ecc71; }
</style>
""", unsafe_allow_html=True)

# --- CACHED EXTRACTION ---
@st.cache_data(show_spinner=False)
def _cached_extract(image_bytes: bytes, filename: str, version: str):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Preprocess
    img = resize_for_ocr(img)
    img = enhance_contrast(img)
    img, _ = deskew(img)
    
    # OCR
    lines = run_ocr(img)
    if not lines: return {"error": "No text detected", "filename": filename}
    
    # Extract
    store, s_conf = extract_store_name(lines)
    date_res = extract_date(lines)
    total_res = extract_total(lines)
    items = extract_items(lines)
    
    # Fallback for Revenue
    total_val = total_res[0] if total_res else f"{calculate_sum_of_items(items):.2f}"
    total_conf = total_res[1] if total_res else 0.5
    
    return {
        "filename": filename,
        "store_name": {"value": store, "confidence": s_conf},
        "date": {"value": date_res[0] if date_res else "Unknown", "confidence": date_res[1] if date_res else 0},
        "total_amount": {"value": total_val, "confidence": total_conf},
        "items": items,
        "n_items": len(items),
        "category": "Groceries" if "walmart" in store.lower() or "spar" in store.lower() else "Retail"
    }

# --- UI MAIN ---
st.title("🌿 Carbon Crunch · AI Receipt Intelligence")
st.caption("v4.5-ULTRA-STABLE (Self-Contained Merge)")

uploaded_files = st.file_uploader("Upload Receipt Images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    results = []
    with st.status("🚀 Intelligence Engine Active...", expanded=True) as status:
        for f in uploaded_files:
            st.write(f"Analyzing {f.name}...")
            res = _cached_extract(f.read(), f.name, "v4.5")
            results.append(res)
        status.update(label="✅ Extraction Complete!", state="complete")
    
    # Dashboard
    summary = generate_summary(results)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue/Spend", f"${summary['total_spend']}")
    c2.metric("Total Items", sum(r.get('n_items', 0) for r in results))
    c3.metric("Receipts Processed", summary['total_transactions'])
    
    # Details
    st.divider()
    for r in results:
        with st.expander(f"📄 {r['filename']} — {r['store_name']['value']} (${r['total_amount']['value']})"):
            col_a, col_b = st.columns(2)
            col_a.write(f"**Store:** {r['store_name']['value']}")
            col_a.write(f"**Date:** {r['date']['value']}")
            col_b.write(f"**Total:** ${r['total_amount']['value']}")
            col_b.write(f"**Items:** {r['n_items']}")
Pass
