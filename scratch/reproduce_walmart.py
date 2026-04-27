from src.extractor import extract_total, extract_items, extract_store_name, extract_date

# Mock lines from the Walmart screenshot
mock_lines = [
    {"text": "Walmart", "confidence": 0.99, "bbox": [[100, 10], [200, 10], [200, 30], [100, 30]], "y_center": 20},
    {"text": "Save money. Live better.", "confidence": 0.95, "y_center": 40},
    {"text": "ST# 01965 OP# 002031 TE# 20 TR# 05145", "confidence": 0.90, "y_center": 100},
    {"text": "SW HRO FGHTR 063060940732 6.94 T", "confidence": 0.92, "y_center": 120, "bbox": [[50, 110], [400, 110], [400, 130], [50, 130]]},
    {"text": "SUBTOTAL", "confidence": 0.95, "y_center": 140, "bbox": [[50, 135], [150, 135], [150, 145], [50, 145]]},
    {"text": "6.94", "confidence": 0.98, "y_center": 140, "bbox": [[350, 135], [400, 135], [400, 145], [350, 145]]},
    {"text": "TAX 1 7.000 % 0.49", "confidence": 0.94, "y_center": 160},
    {"text": "TOTAL", "confidence": 0.99, "y_center": 180, "bbox": [[50, 175], [150, 175], [150, 185], [50, 185]]},
    {"text": "7.43", "confidence": 0.99, "y_center": 180, "bbox": [[350, 175], [400, 175], [400, 185], [350, 185]]},
    {"text": "DEBIT TEND 7.43", "confidence": 0.95, "y_center": 200},
    {"text": "CHANGE DUE 0.00", "confidence": 0.95, "y_center": 220},
    {"text": "12/08/15", "confidence": 0.95, "y_center": 260},
]

print("--- Extraction Test ---")
store = extract_store_name(mock_lines)
print(f"Store: {store}")

date = extract_date(mock_lines)
print(f"Date: {date}")

total = extract_total(mock_lines)
print(f"Total: {total}")

items = extract_items(mock_lines)
print(f"Items: {len(items)}")
for it in items:
    print(f"  {it}")
