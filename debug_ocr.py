"""Quick diagnostic: run with venv/Scripts/python.exe debug_ocr.py"""
from pathlib import Path
from preprocess import preprocess
from ocr import run_ocr
from extractor import (
    extract_total, extract_items, extract_store_name, extract_date,
    _merge_lines_into_rows, DECIMAL_CURRENCY_RE, _UPC_RE
)

# Use the smallest / simplest receipt for a quick test
img_path = Path("data/receipts/21.jpg")
print("Testing on:", img_path)

img, metrics = preprocess(img_path, return_metrics=True)
lines = run_ocr(img)
print(f"OCR lines detected: {len(lines)}")
print("--- ALL RAW OCR LINES ---")
for i, l in enumerate(lines):
    print(f"  [{i:02d}] conf={l['confidence']:.2f} | y={l['y_center']:.0f} | {repr(l['text'])}")

print()
merged = _merge_lines_into_rows(lines)
print("--- MERGED ROWS ---")
for i, m in enumerate(merged):
    stripped = _UPC_RE.sub(" ", m["text"]).strip()
    prices = [g.group(1) for g in DECIMAL_CURRENCY_RE.finditer(stripped)]
    print(f"  [{i:02d}] y={m['y_center']:.0f} | prices={prices} | {repr(m['text'])}")

print()
total = extract_total(lines)
print("extract_total  =>", total)

items = extract_items(lines)
print(f"extract_items  => {len(items)} items")
for it in items:
    print("  ", it)

store = extract_store_name(lines)
print("store          =>", store)

date = extract_date(lines)
print("date           =>", date)
