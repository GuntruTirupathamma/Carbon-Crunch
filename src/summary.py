"""
Financial summary generator (Stage 4 of the pipeline).

Aggregates structured per-receipt outputs into the macro-level metrics
shown in the Carbon Crunch blueprint:

  - Total spend
  - Daily average
  - Top category
  - Number of transactions / extracted / validated fields
  - Per-store spend
  - Per-category spend
  - Daily spend (last 7 days)
  - Overall confidence score (mean across all receipts)
  - Receipts flagged for human review
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional


def _to_float(amount) -> Optional[float]:
    if amount is None:
        return None
    try:
        return float(str(amount).replace(",", ""))
    except (ValueError, TypeError):
        return None


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    s = str(date_str).strip()
    formats = [
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
        "%d/%m/%y", "%d-%m-%y", "%d.%m.%y",
        "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
        "%m/%d/%Y", "%m-%d-%Y",
        "%d %b %Y", "%d %B %Y",
        "%b %d, %Y", "%B %d, %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def generate_summary(results: List[Dict]) -> Dict:
    """
    Build the macro-level financial summary from per-receipt extraction
    results.

    Returns the JSON shape expected by the dashboard.
    """
    successes = [r for r in results if "error" not in r]
    failures  = [r for r in results if "error" in r]

    per_store_total = defaultdict(float)
    per_store_count = defaultdict(int)
    per_category    = defaultdict(float)
    per_category_count = defaultdict(int)
    daily_spend     = defaultdict(float)
    valid_amounts   = []
    parsed_dates    = []
    low_conf_count  = 0
    overall_confs   = []
    extracted_fields = 0
    validated_fields = 0

    for r in successes:
        # Unwrap nested {value, confidence} dicts produced by main.py
        total_block  = r.get("total_amount") or {}
        store_block  = r.get("store_name") or {}
        date_block   = r.get("date") or {}

        total_val = total_block.get("value") if isinstance(total_block, dict) else total_block
        store_val = store_block.get("value") if isinstance(store_block, dict) else store_block
        date_val  = date_block.get("value")  if isinstance(date_block,  dict) else date_block

        total_conf_val = total_block.get("confidence", 0) if isinstance(total_block, dict) else 0
        store_conf_val = store_block.get("confidence", 0) if isinstance(store_block, dict) else 0
        date_conf_val  = date_block.get("confidence", 0)  if isinstance(date_block,  dict) else 0

        amt = _to_float(total_val)
        if amt is None or amt <= 0:
            continue

        store    = (store_val or "Unknown").strip() or "Unknown"
        category = (r.get("category") or "Uncategorized").strip() or "Uncategorized"

        valid_amounts.append(amt)
        per_store_total[store] += amt
        per_store_count[store] += 1
        per_category[category] += amt
        per_category_count[category] += 1

        # Daily aggregation
        dt = _parse_date(date_val)
        if dt:
            parsed_dates.append(dt)
            day_key = dt.strftime("%Y-%m-%d")
            daily_spend[day_key] += amt

        # Overall confidence — average of the three main field confidences
        field_confs = [c for c in [store_conf_val, date_conf_val, total_conf_val] if c > 0]
        if field_confs:
            overall_confs.append(sum(field_confs) / len(field_confs))

        # Field extraction / validation counts
        for conf_val, raw_val in [
            (store_conf_val, store_val),
            (date_conf_val,  date_val),
            (total_conf_val, total_val),
        ]:
            if raw_val:
                extracted_fields += 1
            if conf_val >= 0.7:
                validated_fields += 1
        # Items also count
        extracted_fields += r.get("n_items", 0)

        if r.get("low_confidence_flags"):
            low_conf_count += 1

    total_spend = sum(valid_amounts)
    n_txn = len(valid_amounts)

    # Daily average — based on actual span of receipt dates if available
    if parsed_dates and total_spend > 0:
        date_span_days = (max(parsed_dates) - min(parsed_dates)).days + 1
        daily_average = total_spend / max(date_span_days, 1)
    else:
        daily_average = total_spend / max(n_txn, 1)

    # Top category by spend
    top_category = "—"
    if per_category:
        top_category = max(per_category.items(), key=lambda x: x[1])[0]

    # Last 7 days of spend (for the bar chart)
    last_7_days = []
    today = datetime.now()
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        key = day.strftime("%Y-%m-%d")
        last_7_days.append({
            "date": key,
            "weekday": day.strftime("%a"),
            "amount": round(daily_spend.get(key, 0.0), 2),
        })

    return {
        "total_transactions_processed": len(results),
        "successful_extractions": len(successes),
        "failed_extractions":     len(failures),

        "total_spend":    round(total_spend, 2),
        "daily_average":  round(daily_average, 2),
        "average_transaction": round(total_spend / n_txn, 2) if n_txn else 0.0,
        "max_single_transaction": round(max(valid_amounts), 2) if valid_amounts else 0.0,
        "min_single_transaction": round(min(valid_amounts), 2) if valid_amounts else 0.0,

        "top_category": top_category,
        "spend_per_category": {
            cat: {
                "total": round(per_category[cat], 2),
                "transactions": per_category_count[cat],
            }
            for cat in sorted(per_category, key=lambda c: -per_category[c])
        },

        "spend_per_store": {
            store: {
                "total": round(per_store_total[store], 2),
                "transactions": per_store_count[store],
                "average": round(per_store_total[store] / per_store_count[store], 2),
            }
            for store in sorted(per_store_total, key=lambda s: -per_store_total[s])
        },

        "daily_spend_last_7_days": last_7_days,

        "extracted_fields_count":  extracted_fields,
        "validated_fields_count":  validated_fields,
        "overall_confidence":      round(sum(overall_confs)/len(overall_confs), 3)
                                     if overall_confs else 0.0,
        "low_confidence_receipt_count": low_conf_count,
    }
