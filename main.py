"""
Carbon Crunch — End-to-end OCR receipt extraction pipeline.

Reads receipt images from data/receipts/, runs preprocessing → OCR →
field extraction → confidence scoring, and writes:

  outputs/json/<image>.json      — structured per-receipt output
  outputs/summary.json           — portfolio-level financial summary
  outputs/preview/<image>.png    — before/after preprocessing samples

Usage:
    python main.py
    python main.py --input my_receipts/ --output my_outputs/
    python main.py --no-preview          # skip preview images
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

from preprocess import preprocess, save_preview
from ocr import run_ocr, get_full_text, get_average_confidence
from extractor import (
    extract_date, extract_store_name, extract_total, extract_items,
    detect_category, detect_currency, calculate_sum_of_items,
)
from confidence import (
    adjust_confidence, collect_low_confidence_flags,
)
from summary import generate_summary


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")


SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")


def process_receipt(image_path: Path,
                    save_preview_to: Path | None = None) -> Dict:
    """
    Full single-receipt pipeline.

    Returns a dict matching the assignment's required JSON schema, with
    confidence-aware fields and a low_confidence_flags helper list.
    """
    # 1. Preprocess
    img, prep_metrics = preprocess(image_path, return_metrics=True)

    # 2. OCR
    lines = run_ocr(img)
    if not lines:
        return {
            "file": image_path.name,
            "error": "OCR returned no text — image may be blank or unreadable",
            "preprocessing_metrics": prep_metrics,
        }

    raw_text = get_full_text(lines)
    avg_ocr_conf = get_average_confidence(lines)

    # 3. Extract fields
    store_name, store_raw_conf = extract_store_name(lines)
    date_result = extract_date(lines)
    total_result = extract_total(lines)
    items_raw = extract_items(lines)

    # Detect category and currency (needed for financial summary)
    category, category_conf = detect_category(store_name or "", raw_text)
    currency_symbol, currency_code = detect_currency(lines)

    # 4. Apply field-level confidence scoring (validation factor)
    date_value = date_result[0] if date_result else None
    date_raw_conf = date_result[1] if date_result else 0.0
    date_conf = adjust_confidence("date", date_value, date_raw_conf)

    total_value = total_result[0] if total_result else None
    total_raw_conf = total_result[1] if total_result else 0.0

    # Fallback to sum of items if total is missing
    if not total_value and items_raw:
        sum_items = calculate_sum_of_items(items_raw)
        if sum_items > 0:
            total_value = f"{sum_items:.2f}"
            total_raw_conf = 0.50  # low confidence fallback

    total_conf = adjust_confidence("total_amount", total_value, total_raw_conf)

    store_conf = adjust_confidence("store_name", store_name, store_raw_conf)

    items = []
    for it in items_raw:
        price_conf = adjust_confidence("price", it["price"], it["confidence"])
        items.append({
            "name": it["description"],
            "price": {
                "value": it["price"],
                "confidence": price_conf,
            },
        })

    # 5. Reliability flags
    low_conf_flags = collect_low_confidence_flags({
        "store_name":   store_conf,
        "date":         date_conf,
        "total_amount": total_conf,
    })

    # 6. Optional preview
    if save_preview_to is not None:
        try:
            save_preview(image_path, img, save_preview_to)
        except Exception as exc:
            log.warning("Preview save failed for %s: %s", image_path.name, exc)

    # 7. Build structured output
    return {
        "file": image_path.name,
        "store_name": {
            "value": store_name,
            "confidence": store_conf,
        },
        "date": {
            "value": date_value,
            "confidence": date_conf,
        },
        "items": items,
        "n_items": len(items),
        "total_amount": {
            "value": total_value,
            "confidence": total_conf,
        },
        "category": category,
        "category_confidence": round(category_conf, 3),
        "currency": {
            "symbol": currency_symbol,
            "code": currency_code,
        },
        "low_confidence_flags": low_conf_flags,
        "metadata": {
            "ocr_lines_detected": len(lines),
            "average_ocr_confidence": round(avg_ocr_conf, 3),
            "preprocessing": {
                "blur_score":       round(prep_metrics["blur_score"], 2),
                "brightness":       round(prep_metrics["brightness"], 2),
                "skew_angle":       round(prep_metrics["skew_angle"], 2),
                "original_shape":   list(prep_metrics["original_shape"]),
                "processed_shape":  list(prep_metrics["processed_shape"]),
            },
        },
        "raw_ocr_text": raw_text,
    }


def run_pipeline(input_dir: Path,
                 output_dir: Path,
                 save_previews: bool = True,
                 max_previews: int = 5) -> List[Dict]:
    """Iterate over all images, run the pipeline, write outputs."""
    json_dir = output_dir / "json"
    preview_dir = output_dir / "preview"
    json_dir.mkdir(parents=True, exist_ok=True)
    if save_previews:
        preview_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    )
    if not images:
        log.warning("No images found in %s", input_dir)
        return []

    log.info("Found %d images. Starting pipeline...", len(images))
    results: List[Dict] = []
    t0 = time.time()

    for i, img_path in enumerate(images, start=1):
        preview_target = (
            preview_dir / f"{img_path.stem}_preview.png"
            if save_previews and i <= max_previews
            else None
        )
        try:
            result = process_receipt(img_path, save_preview_to=preview_target)
            json_path = json_dir / f"{img_path.stem}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            results.append(result)
            status = "OK"
            if "error" in result:
                status = "ERR"
            elif result.get("low_confidence_flags"):
                status = f"LOW({','.join(result['low_confidence_flags'])})"
            log.info("[%d/%d] %s — %s", i, len(images), img_path.name, status)
        except Exception as exc:
            log.error("[%d/%d] %s FAILED: %s", i, len(images), img_path.name, exc)
            err = {"file": img_path.name, "error": str(exc)}
            with open(json_dir / f"{img_path.stem}.json", "w") as f:
                json.dump(err, f, indent=2)
            results.append(err)

    elapsed = time.time() - t0
    log.info("Processed %d images in %.1fs (%.2fs/img)",
             len(images), elapsed, elapsed / len(images))

    # Generate financial summary
    summary = generate_summary(results)
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary written to %s", summary_path)

    # Console summary
    print("\n" + "=" * 60)
    print("FINANCIAL SUMMARY")
    print("=" * 60)
    print(f"  Receipts processed   : {summary['total_transactions_processed']}")
    print(f"  Successful           : {summary['successful_extractions']}")
    print(f"  Failed               : {summary['failed_extractions']}")
    print(f"  Total spend          : {summary['total_spend']:.2f}")
    print(f"  Average transaction  : {summary['average_transaction']:.2f}")
    print(f"  Stores detected      : {len(summary['spend_per_store'])}")
    print(f"  Low-confidence flagged: {summary['low_confidence_receipt_count']}")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Carbon Crunch — OCR receipt extraction pipeline"
    )
    parser.add_argument(
        "--input", type=Path,
        default=Path(r"C:\Users\guntr\OneDrive\Documents\AI-OCR dataset\AI-OCR dataset"),
        help="Folder containing receipt images",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs"),
                        help="Folder to write JSON + summary outputs")
    parser.add_argument("--no-preview", action="store_true",
                        help="Skip writing before/after preview images")
    parser.add_argument("--max-previews", type=int, default=5,
                        help="Max preview images to save")
    args = parser.parse_args()

    if not args.input.exists():
        log.error("Input directory does not exist: %s", args.input)
        sys.exit(1)

    run_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        save_previews=not args.no_preview,
        max_previews=args.max_previews,
    )


if __name__ == "__main__":
    main()
