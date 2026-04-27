"""
Carbon Crunch — AI Receipt Intelligence
=========================================
Production-quality Streamlit app. Run with:
    streamlit run app.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.preprocess import (
    enhance_contrast, denoise, deskew, resize_for_ocr,
    estimate_blur, to_grayscale,
)
from src.ocr import run_ocr, get_average_confidence
from src.extractor import (
    extract_date, extract_store_name, extract_total, extract_items,
    extract_subtotal, extract_tax, detect_currency, detect_category,
    calculate_sum_of_items,
)
from src.summary import generate_summary
from src.confidence import adjust_confidence


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Carbon Crunch · AI Receipt Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CARBON_KG_PER_UNIT = 0.041   # rough industry-average kg CO₂e per ₹/$


# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM (CSS)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@500;600&display=swap');

/* ━━━ TOKENS ━━━ */
:root {
    --bg-0: #050a08;
    --bg-1: #0a1410;
    --bg-2: #0f1d18;
    --line: rgba(0, 200, 150, 0.12);
    --line-strong: rgba(0, 200, 150, 0.28);
    --accent: #00c896;
    --accent-2: #4ade80;
    --accent-glow: rgba(0, 200, 150, 0.4);
    --text-0: #f0fdf4;
    --text-1: #d1fae5;
    --text-2: #94d4b3;
    --text-3: #6b8a7c;
    --text-4: #4b6359;
    --danger: #ef4444;
    --warn: #f59e0b;
}

/* ━━━ RESET ━━━ */
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
.block-container {
    padding: 1rem 2rem 4rem !important;
    max-width: 1280px !important;
}
/* Kill auto-anchor link icons */
.stMarkdown a[href^="#"], h1 a, h2 a, h3 a, h4 a,
[data-testid="stHeaderActionElements"], [class*="StyledLinkIcon"] {
    display: none !important; visibility: hidden !important;
}
.stMarkdown p { margin: 0 !important; }
code, pre { font-family: 'JetBrains Mono', monospace !important; }

/* ━━━ KEYFRAMES ━━━ */
@keyframes pulse  { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
@keyframes glow   { 0%,100% { box-shadow: 0 0 20px rgba(0,200,150,0.25); }
                    50%      { box-shadow: 0 0 32px rgba(0,200,150,0.45); } }
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); }
                    to   { opacity: 1; transform: translateY(0); } }
@keyframes shimmer{ 0% { background-position: 200% 0; }
                    100% { background-position: -200% 0; } }
@keyframes growBar{ from { width: 0%; } }
@keyframes float  { 0%,100% { transform: translateY(0); }
                    50%      { transform: translateY(-6px); } }

/* ━━━ NAVBAR ━━━ */
.nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.85rem 1.3rem;
    background: rgba(8, 18, 14, 0.7);
    backdrop-filter: blur(24px);
    border: 1px solid var(--line);
    border-radius: 14px;
    margin-bottom: 3rem;
    animation: fadeIn 0.5s ease;
}
.brand { display: flex; align-items: center; gap: 0.75rem; }
.brand-mark {
    width: 38px; height: 38px;
    border-radius: 11px;
    background: linear-gradient(135deg, var(--accent), #00a884);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 16px var(--accent-glow);
}
.brand-mark svg { width: 20px; height: 20px; }
.brand-text {
    line-height: 1.1;
}
.brand-text .name {
    font-size: 1rem; font-weight: 700; color: var(--text-0);
    letter-spacing: -0.01em;
}
.brand-text .subtitle {
    font-size: 0.72rem; color: var(--accent-2);
    font-weight: 500; opacity: 0.85; margin-top: 2px;
}
.nav-pill {
    background: rgba(0, 200, 150, 0.1);
    border: 1px solid var(--line-strong);
    color: var(--accent-2);
    padding: 0.32rem 0.85rem; border-radius: 999px;
    font-size: 0.66rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
}
.nav-pill.muted {
    background: rgba(255,255,255,0.04);
    border-color: rgba(255,255,255,0.08);
    color: var(--text-3);
}
.status-dot {
    width: 7px; height: 7px; background: var(--accent); border-radius: 50%;
    box-shadow: 0 0 8px var(--accent);
    animation: pulse 2s ease-in-out infinite;
    display: inline-block; margin-right: 0.4rem; vertical-align: middle;
}
.nav-meta { display: flex; gap: 0.5rem; align-items: center; }

/* ━━━ HERO ━━━ */
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(0, 200, 150, 0.1);
    border: 1px solid var(--line-strong);
    color: var(--accent-2);
    padding: 0.4rem 1rem; border-radius: 999px;
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase;
    margin-bottom: 1.6rem;
    animation: fadeIn 0.6s ease;
}
.hero-h1 {
    font-size: 3.2rem; font-weight: 800; line-height: 1.08;
    letter-spacing: -0.04em; color: var(--text-0);
    margin: 0 0 1.4rem 0;
    animation: fadeIn 0.7s ease;
}
.hero-h1 .accent {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    color: var(--text-2); font-size: 1.05rem; line-height: 1.65;
    margin: 0 0 2rem 0; max-width: 480px; opacity: 0.9;
    animation: fadeIn 0.8s ease;
}
.feat-tags {
    display: flex; gap: 0.6rem; flex-wrap: wrap;
    animation: fadeIn 0.9s ease;
}
.feat-tag {
    background: rgba(8, 18, 14, 0.6);
    border: 1px solid var(--line);
    color: var(--text-1);
    padding: 0.5rem 0.95rem; border-radius: 10px;
    font-size: 0.78rem; font-weight: 500;
    display: inline-flex; align-items: center; gap: 0.45rem;
    transition: all 0.2s;
}
.feat-tag:hover {
    border-color: var(--line-strong);
    background: rgba(0, 200, 150, 0.06);
    transform: translateY(-1px);
}
.feat-tag .label { color: var(--accent-2); font-weight: 600; }
.feat-tag svg { width: 14px; height: 14px; color: var(--accent-2); }

.hero-art {
    position: relative; height: 380px; width: 100%;
    display: flex; align-items: center; justify-content: center;
    animation: fadeIn 1s ease;
}

/* ━━━ SECTION HEADERS ━━━ */
.section-eyebrow {
    color: var(--accent-2); font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.16em; text-transform: uppercase;
    text-align: center; margin-bottom: 0.8rem;
}
.section-title {
    color: var(--text-0); font-size: 1.95rem; font-weight: 700;
    letter-spacing: -0.02em; text-align: center;
    margin: 0 0 0.7rem 0;
}
.section-sub {
    color: var(--text-2); font-size: 0.95rem;
    text-align: center; margin: 0 auto 2.2rem auto;
    max-width: 540px; opacity: 0.85; line-height: 1.6;
}

/* ━━━ FILE UPLOADER ━━━ */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(8, 18, 14, 0.55) !important;
    border: 2px dashed var(--line-strong) !important;
    border-radius: 18px !important;
    padding: 3rem 2rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important;
    background: rgba(0, 200, 150, 0.05) !important;
    box-shadow: 0 0 40px rgba(0, 200, 150, 0.18);
    transform: scale(1.005);
}
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] div { color: var(--text-2) !important; }

/* Format badges below uploader */
.format-badges {
    display: flex; justify-content: center; gap: 0.5rem;
    margin-top: 1rem; flex-wrap: wrap;
}
.format-badge {
    background: rgba(0, 200, 150, 0.06);
    border: 1px solid var(--line);
    color: var(--accent-2);
    padding: 0.3rem 0.75rem; border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.05em;
}

/* ━━━ BUTTONS ━━━ */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, var(--accent), #00a884) !important;
    color: #052218 !important;
    border: none !important; border-radius: 11px !important;
    padding: 0.78rem 1.5rem !important;
    font-weight: 700 !important; font-size: 0.9rem !important;
    box-shadow: 0 4px 16px rgba(0, 200, 150, 0.3) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 10px 28px rgba(0, 200, 150, 0.45) !important;
}

/* ━━━ RESULT CARDS (GLASSMORPHISM) ━━━ */
.result-card {
    background: linear-gradient(135deg,
                rgba(0, 200, 150, 0.04),
                rgba(8, 18, 14, 0.65));
    border: 1px solid var(--line-strong);
    border-radius: 16px;
    padding: 1.8rem;
    backdrop-filter: blur(24px);
    position: relative;
    overflow: hidden;
    animation: fadeIn 0.5s ease;
}
.result-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-glow), transparent);
}
.result-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 1.5rem; padding-bottom: 1rem;
    border-bottom: 1px solid var(--line);
}
.result-title {
    color: var(--text-0); font-size: 1.1rem; font-weight: 700;
    letter-spacing: -0.01em;
}
.result-title .icon {
    display: inline-flex; vertical-align: middle; margin-right: 0.5rem;
    color: var(--accent-2);
}
.confidence-pill {
    display: inline-flex; align-items: center; gap: 0.45rem;
    padding: 0.35rem 0.85rem;
    border-radius: 999px; font-size: 0.72rem; font-weight: 700;
    border: 1px solid var(--line-strong);
    background: rgba(0, 200, 150, 0.1);
    color: var(--accent-2);
    font-family: 'JetBrains Mono', monospace;
}
.confidence-pill .dot {
    width: 6px; height: 6px; background: var(--accent);
    border-radius: 50%; box-shadow: 0 0 8px var(--accent);
    animation: pulse 2s ease-in-out infinite;
}

/* ━━━ FIELD ROWS (with animated confidence bars) ━━━ */
.field {
    padding: 0.95rem 0;
    border-bottom: 1px solid rgba(0, 200, 150, 0.06);
    animation: fadeIn 0.4s ease;
}
.field:last-of-type { border-bottom: none; }
.field-row {
    display: flex; align-items: center; justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.5rem;
}
.field-label {
    color: var(--text-3); font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.16em; text-transform: uppercase;
}
.field-value {
    color: var(--text-0); font-size: 1rem; font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    text-align: right;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    max-width: 260px;
}

.conf-bar-container {
    display: flex; align-items: center; gap: 0.7rem;
    margin-top: 0.4rem;
}
.conf-bar {
    flex: 1; height: 5px;
    background: rgba(255,255,255,0.05);
    border-radius: 999px; overflow: hidden;
}
.conf-bar-fill {
    height: 100%; border-radius: 999px;
    animation: growBar 0.8s ease-out;
    transition: width 0.3s;
}
.conf-bar-fill.high { background: linear-gradient(90deg, var(--accent), var(--accent-2)); }
.conf-bar-fill.mid  { background: linear-gradient(90deg, var(--warn), #fbbf24); }
.conf-bar-fill.low  { background: linear-gradient(90deg, var(--danger), #f87171); }
.conf-pct {
    color: var(--text-1); font-size: 0.75rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    min-width: 40px; text-align: right;
}

/* ━━━ CO₂ HERO (THE PRIMARY INSIGHT — Carbon Crunch's whole reason for being) ━━━ */
.co2-hero {
    background:
      radial-gradient(circle at 20% 30%, rgba(0, 200, 150, 0.2) 0%, transparent 60%),
      linear-gradient(135deg, rgba(0, 200, 150, 0.18) 0%, rgba(74, 222, 128, 0.06) 100%);
    border: 1px solid rgba(0, 200, 150, 0.45);
    border-radius: 16px;
    padding: 1.8rem 1.5rem;
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
    animation: co2Glow 3s ease-in-out infinite;
}
@keyframes co2Glow {
    0%,100% { box-shadow: 0 0 28px rgba(0,200,150,0.25), inset 0 0 0 1px rgba(0,200,150,0.15); }
    50%     { box-shadow: 0 0 48px rgba(0,200,150,0.45), inset 0 0 0 1px rgba(0,200,150,0.3); }
}
.co2-hero::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
}
.co2-hero::after {
    content: ''; position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(0,200,150,0.18), transparent 70%);
    pointer-events: none;
    animation: float 5s ease-in-out infinite;
}
.co2-hero-row {
    display: flex; align-items: center; gap: 1.2rem;
    position: relative; z-index: 1;
}
.co2-hero-icon {
    width: 64px; height: 64px; flex-shrink: 0;
    background: linear-gradient(135deg, rgba(0, 200, 150, 0.25), rgba(0, 200, 150, 0.1));
    border: 1px solid rgba(0, 200, 150, 0.4);
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 8px 24px rgba(0, 200, 150, 0.25);
}
.co2-hero-icon svg { width: 32px; height: 32px; color: var(--accent-2); }
.co2-hero-content { flex: 1; }
.co2-hero-label {
    color: var(--accent-2); font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.18em; text-transform: uppercase;
    margin-bottom: 0.4rem;
    display: flex; align-items: center; justify-content: space-between;
}
.co2-hero-label .factor-tag {
    background: rgba(0, 200, 150, 0.1);
    border: 1px solid rgba(0, 200, 150, 0.25);
    color: var(--accent-2);
    padding: 0.15rem 0.55rem; border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; font-weight: 600;
    letter-spacing: 0.04em; text-transform: none;
}
.co2-hero-value {
    color: var(--text-0);
    font-size: 3rem; font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.04em;
    line-height: 1;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-2) 60%, var(--accent) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.co2-hero-value .unit {
    font-size: 1.1rem; opacity: 0.7;
    color: var(--accent-2); -webkit-text-fill-color: var(--accent-2);
    margin-left: 0.45rem; font-weight: 500;
}
.co2-hero-context {
    margin-top: 0.9rem;
    color: var(--text-3); font-size: 0.78rem; line-height: 1.5;
    position: relative; z-index: 1;
}
.co2-hero-context strong {
    color: var(--text-1); font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

/* ━━━ MONEY SUPPORT CARD (audit input — secondary) ━━━ */
.money-input {
    background: rgba(8, 18, 14, 0.6);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1.4rem;
}
.money-input-label {
    color: var(--text-3); font-size: 0.66rem; font-weight: 700;
    letter-spacing: 0.16em; text-transform: uppercase;
    margin-bottom: 0.7rem;
    display: flex; align-items: center; justify-content: space-between;
}
.money-input-label .currency-tag {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    color: var(--text-2);
    padding: 0.15rem 0.55rem; border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; font-weight: 600;
    letter-spacing: 0.04em;
}
.money-input-value {
    color: var(--text-1); font-size: 1.6rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.02em; line-height: 1;
}
.money-input-value .symbol {
    color: var(--text-3); font-size: 1.1rem; margin-right: 0.25rem;
    font-weight: 500;
}
.money-input-meta {
    margin-top: 0.7rem;
    color: var(--text-3); font-size: 0.75rem;
    display: flex; gap: 1.2rem; flex-wrap: wrap;
}
.money-input-meta .key { color: var(--text-3); font-weight: 500; }
.money-input-meta .val {
    color: var(--text-1); font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    margin-left: 0.3rem;
}

/* ━━━ MONEY BREAKDOWN MINI-TABLE ━━━ */
.breakdown {
    background: rgba(8, 18, 14, 0.5);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1.4rem;
}
.breakdown-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(0, 200, 150, 0.05);
}
.breakdown-row:last-child {
    border-bottom: none; padding-top: 0.7rem;
    margin-top: 0.2rem; border-top: 1px solid var(--line-strong);
}
.breakdown-label {
    color: var(--text-3); font-size: 0.78rem; font-weight: 500;
}
.breakdown-value {
    color: var(--text-1); font-size: 0.9rem; font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.breakdown-row.is-total .breakdown-label {
    color: var(--accent-2); font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
    font-size: 0.7rem;
}
.breakdown-row.is-total .breakdown-value {
    color: var(--text-0); font-size: 1.05rem; font-weight: 800;
}

/* ━━━ CARBON HIGHLIGHT (with glow) ━━━ */
.carbon-card {
    margin-top: 1.5rem; padding: 1.4rem 1.5rem;
    background: linear-gradient(135deg,
                rgba(0, 200, 150, 0.18) 0%,
                rgba(74, 222, 128, 0.08) 100%);
    border: 1px solid var(--line-strong);
    border-left: 3px solid var(--accent);
    border-radius: 14px;
    display: flex; align-items: center; gap: 1.1rem;
    animation: glow 3s ease-in-out infinite;
    position: relative; overflow: hidden;
}
.carbon-card::after {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(circle at 0% 50%,
                rgba(0, 200, 150, 0.15), transparent 70%);
    pointer-events: none;
}
.carbon-icon {
    width: 48px; height: 48px; flex-shrink: 0;
    background: rgba(0, 200, 150, 0.2);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    z-index: 1;
}
.carbon-icon svg { width: 24px; height: 24px; color: var(--accent-2); }
.carbon-text { flex: 1; z-index: 1; }
.carbon-label {
    color: var(--accent-2); font-size: 0.66rem; font-weight: 700;
    letter-spacing: 0.16em; text-transform: uppercase;
    margin-bottom: 0.25rem;
}
.carbon-value {
    color: var(--text-0); font-size: 1.5rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.02em;
}
.carbon-value .unit {
    color: var(--text-3); font-size: 0.78rem;
    font-weight: 500; margin-left: 0.4rem;
}

/* ━━━ IMAGE PREVIEW ━━━ */
.preview-wrap {
    background: rgba(8, 18, 14, 0.55);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1.2rem;
    position: relative;
    transition: all 0.25s;
    animation: fadeIn 0.5s ease;
}
.preview-wrap:hover {
    border-color: var(--line-strong);
    transform: translateY(-1px);
}
.preview-label {
    color: var(--text-3); font-size: 0.66rem; font-weight: 700;
    letter-spacing: 0.16em; text-transform: uppercase;
    margin-bottom: 0.9rem;
    display: flex; align-items: center; gap: 0.4rem;
}
.preview-label .file { color: var(--text-1); text-transform: none; letter-spacing: 0.01em; }
[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid var(--line);
}

/* ━━━ HOW IT WORKS STEPS ━━━ */
.step {
    background: rgba(8, 18, 14, 0.55);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1.8rem 1.4rem;
    text-align: center;
    height: 100%;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    animation: fadeIn 0.6s ease;
}
.step:hover {
    transform: translateY(-4px);
    border-color: var(--line-strong);
    box-shadow: 0 16px 40px rgba(0, 200, 150, 0.12);
    background: rgba(0, 200, 150, 0.03);
}
.step-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 40px; height: 40px;
    background: linear-gradient(135deg, var(--accent), #00a884);
    color: #052218; font-weight: 800; font-size: 1rem;
    border-radius: 11px;
    margin: 0 auto 1rem auto;
    box-shadow: 0 4px 16px var(--accent-glow);
}
.step-title {
    color: var(--text-0); font-size: 1.05rem; font-weight: 700;
    margin: 0 0 0.5rem 0; letter-spacing: -0.01em;
}
.step-text {
    color: var(--text-2); font-size: 0.88rem; line-height: 1.6;
    opacity: 0.88;
}

/* ━━━ EMPTY STATE ━━━ */
.empty {
    text-align: center; padding: 4rem 2rem;
    animation: fadeIn 0.5s ease;
}
.empty-icon {
    width: 60px; height: 60px;
    margin: 0 auto 1.1rem;
    background: rgba(0, 200, 150, 0.08);
    border: 1px solid var(--line-strong);
    border-radius: 15px;
    display: flex; align-items: center; justify-content: center;
}
.empty-icon svg { width: 28px; height: 28px; color: var(--accent-2); }
.empty-title {
    color: var(--text-1); font-size: 1.05rem; font-weight: 500;
    margin-bottom: 0.4rem;
}
.empty-sub { color: var(--text-3); font-size: 0.82rem; }

/* ━━━ MACRO DASHBOARD (Stage 4) ━━━ */
.macro-banner {
    background: linear-gradient(135deg,
                rgba(0, 200, 150, 0.10) 0%,
                rgba(0, 200, 150, 0.02) 100%);
    border: 1px solid var(--line-strong);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.8rem;
    text-align: center;
}
.macro-banner-eyebrow {
    color: var(--accent-2); font-size: 0.66rem; font-weight: 700;
    letter-spacing: 0.18em; text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.macro-banner-title {
    color: var(--text-0); font-size: 1.5rem; font-weight: 700;
    letter-spacing: -0.02em; margin-bottom: 0.4rem;
}
.macro-banner-sub {
    color: var(--text-2); font-size: 0.88rem; opacity: 0.85;
}

.kpi-card {
    background: rgba(8, 18, 14, 0.6);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1.3rem 1.2rem;
    height: 100%;
    transition: all 0.2s;
}
.kpi-card:hover {
    border-color: var(--line-strong);
    transform: translateY(-2px);
}
.kpi-label {
    color: var(--text-3); font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.16em; text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.kpi-value {
    color: var(--text-0); font-size: 1.7rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.03em; line-height: 1;
}
.kpi-value .symbol {
    color: var(--text-3); font-size: 1.1rem;
    font-weight: 500; margin-right: 0.2rem;
}
.kpi-meta {
    color: var(--text-3); font-size: 0.72rem;
    margin-top: 0.4rem;
}

/* Confidence donut (KPI card variant) */
.donut-wrap {
    display: flex; align-items: center; gap: 1rem;
}
.donut {
    width: 70px; height: 70px;
    border-radius: 50%;
    background: conic-gradient(var(--accent) calc(var(--pct) * 1%),
                                rgba(255,255,255,0.05) 0);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 0 16px rgba(0,200,150,0.2);
}
.donut::before {
    content: ''; width: 54px; height: 54px;
    background: var(--bg-1);
    border-radius: 50%;
    position: absolute;
}
.donut span {
    position: relative; z-index: 1;
    color: var(--accent-2); font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
}
.donut-content { flex: 1; }

/* Categorized transactions table */
.txn-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    background: rgba(8, 18, 14, 0.5);
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
}
.txn-table th {
    background: rgba(0, 200, 150, 0.06);
    color: var(--accent-2);
    font-size: 0.6rem; font-weight: 700;
    letter-spacing: 0.16em; text-transform: uppercase;
    padding: 0.7rem 1rem; text-align: left;
}
.txn-table td {
    padding: 0.7rem 1rem;
    color: var(--text-1); font-size: 0.82rem;
    border-top: 1px solid rgba(0, 200, 150, 0.05);
    font-family: 'JetBrains Mono', monospace;
}
.txn-table tr:hover { background: rgba(0, 200, 150, 0.04); }
.cat-pill {
    display: inline-block;
    padding: 0.18rem 0.55rem;
    border-radius: 999px;
    font-size: 0.66rem; font-weight: 700;
    letter-spacing: 0.04em;
    background: rgba(0,200,150,0.1);
    color: var(--accent-2);
    border: 1px solid rgba(0,200,150,0.25);
    font-family: 'Inter', sans-serif;
}
.txn-status-ok   { color: var(--accent-2); font-weight: 700; }
.txn-status-flag { color: var(--warn); font-weight: 700; }

/* ━━━ FOOTER ━━━ */
.app-footer {
    text-align: center; color: var(--text-4); font-size: 0.78rem;
    padding: 2.5rem 0 0.5rem;
    border-top: 1px solid rgba(0, 200, 150, 0.05);
    margin-top: 4.5rem;
}
.app-footer strong { color: var(--accent-2); font-weight: 600; }

/* ━━━ SPINNER ━━━ */
.stSpinner > div > div { border-top-color: var(--accent) !important; }

/* ━━━ PROGRESS ━━━ */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
    background-size: 200% 100% !important;
    animation: shimmer 2s linear infinite !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CACHED PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _warm_ocr_reader():
    from src.ocr import _get_reader
    return _get_reader()


@st.cache_data(show_spinner=False, max_entries=100)
def _cached_extract(file_bytes: bytes, filename: str) -> dict:
    """Lightweight extraction (no images) — cached by file hash."""
    arr = np.frombuffer(file_bytes, np.uint8)
    img_orig = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_orig is None:
        return {"error": "Cannot decode image"}

    gray = to_grayscale(img_orig)
    blur_score = estimate_blur(gray)

    img = resize_for_ocr(img_orig, target_height=1200, max_height=1800)
    img = enhance_contrast(img)
    if blur_score < 200:
        img = denoise(img)
    img, _ = deskew(img)

    lines = run_ocr(img)
    if not lines:
        return {"error": "No text detected"}

    avg_ocr = get_average_confidence(lines)
    full_text = " ".join(l["text"] for l in lines)
    store_name, store_raw = extract_store_name(lines)
    date_res     = extract_date(lines)
    total_res    = extract_total(lines)
    subtotal_res = extract_subtotal(lines)
    tax_res      = extract_tax(lines)
    items_raw    = extract_items(lines)
    currency_sym, currency_code = detect_currency(lines)
    category, category_conf = detect_category(store_name, full_text)

    date_v     = date_res[0]     if date_res     else None
    total_v    = total_res[0]    if total_res    else None
    total_raw_conf = total_res[1] if total_res else 0.0

    # Fallback to sum of items if total is missing
    if not total_v and items_raw:
        sum_val = calculate_sum_of_items(items_raw)
        if sum_val > 0:
            total_v = f"{sum_val:.2f}"
            total_raw_conf = 0.50  # low confidence fallback
    
    subtotal_v = subtotal_res[0] if subtotal_res else None
    tax_v      = tax_res[0]      if tax_res      else None

    store_c = adjust_confidence("store_name", store_name, store_raw)
    date_c  = adjust_confidence("date", date_v, date_res[1] if date_res else 0)
    total_c = adjust_confidence("total_amount", total_v, total_raw_conf)
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
        "subtotal":   subtotal_v,
        "tax":        tax_v,
        "currency_symbol": currency_sym,
        "currency_code":   currency_code,
        "category":        category,
        "category_conf":   category_conf,
        "store_conf": store_c,
        "date_conf":  date_c,
        "total_conf": total_c,
        "overall_conf": overall,
        "avg_ocr": avg_ocr,
        "carbon_kg": round(carbon, 3),
        "n_items": len(items_raw),
        "items": items_raw[:30],
        "low_confidence_flags": [
            f for f, c in [("store_name", store_c), ("date", date_c),
                            ("total_amount", total_c)] if c < 0.7
        ],
    }


def process_file(file_bytes: bytes, filename: str) -> dict:
    result = _cached_extract(file_bytes, filename)
    if "error" in result:
        return result
    arr = np.frombuffer(file_bytes, np.uint8)
    img_orig = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return {**result, "_original_image": img_orig}


# Warm OCR reader (cached, non-blocking — failures don't crash the page)
try:
    _warm_ocr_reader()
except Exception:
    pass

# ── Cache-version guard ──────────────────────────────────────────────────────
# Bump this string whenever extraction logic changes so stale cached results
# from previous code versions are automatically discarded.
_CACHE_VERSION = "v2.6-revenue-fix"

if st.session_state.get("_cache_version") != _CACHE_VERSION:
    st.cache_data.clear()
    st.session_state.clear()
    st.session_state["_cache_version"] = _CACHE_VERSION

# Drop stale session results from an older schema (prevents KeyError on
# missing keys like currency_symbol / subtotal / tax after a code update).
if "results" in st.session_state:
    try:
        _r0 = st.session_state["results"][0]
        if "error" not in _r0 and not all(k in _r0 for k in
                ("currency_symbol", "subtotal", "tax", "category")):
            st.session_state.pop("results", None)
            st.session_state.pop("file_signature", None)
            st.cache_data.clear()
    except (IndexError, KeyError, TypeError):
        st.session_state.pop("results", None)
        st.session_state.pop("file_signature", None)
        st.cache_data.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def conf_class(s: float) -> str:
    return "high" if s >= 0.7 else ("mid" if s >= 0.5 else "low")


def section_head(title: str, count: str = "") -> str:
    cnt = (f'<span style="color:var(--text-3);font-size:0.78rem;'
           f'font-weight:500;margin-left:0.4rem">· {count}</span>') if count else ""
    return (
        f'<div style="display:flex;align-items:center;gap:0.6rem;'
        f'margin:1.6rem 0 1rem 0">'
        f'<span style="width:3px;height:16px;'
        f'background:linear-gradient(180deg,#10b981,#14b8a6);'
        f'border-radius:2px"></span>'
        f'<span style="color:#ecfdf5;font-size:0.95rem;font-weight:600;'
        f'letter-spacing:-0.01em">{title}</span>{cnt}</div>'
    )


def render_field(label: str, value: str, conf: float | None = None) -> str:
    val = value if value not in (None, "") else "—"
    if conf is None:
        return (
            f'<div class="field">'
            f'<div class="field-row">'
            f'<span class="field-label">{label}</span>'
            f'<span class="field-value">{val}</span>'
            f'</div></div>'
        )
    pct = int(conf * 100)
    cls = conf_class(conf)
    return (
        f'<div class="field">'
        f'<div class="field-row">'
        f'<span class="field-label">{label}</span>'
        f'<span class="field-value">{val}</span>'
        f'</div>'
        f'<div class="conf-bar-container">'
        f'<div class="conf-bar"><div class="conf-bar-fill {cls}" '
        f'style="width:{pct}%"></div></div>'
        f'<span class="conf-pct">{pct}%</span>'
        f'</div></div>'
    )


# ═══════════════════════════════════════════════════════════════════════════════
# NAVBAR
# ═══════════════════════════════════════════════════════════════════════════════
LOGO_SVG = """
<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M12 2C8 6 4 9 4 14C4 18.4 7.6 22 12 22C16.4 22 20 18.4 20 14C20 9 16 6 12 2Z"
        fill="#052218" stroke="#ecfdf5" stroke-width="1.6" stroke-linejoin="round"/>
  <path d="M12 11C10 13 8 15 8 17" stroke="#4ade80" stroke-width="1.6"
        stroke-linecap="round"/>
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
# 1. HERO
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
          <rect x="0" y="0" width="150" height="74" rx="12" fill="none" stroke="rgba(0,200,150,0.15)" stroke-width="3" filter="blur(4px)"/>
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
# 2. UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div style="margin-top:3.5rem"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-eyebrow">◇ Step 1 — Upload</div>
<div class="section-title">Drop receipts or browse</div>
<div class="section-sub">
  Process one or many at once. Images stay on your machine —
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

    if "results" in st.session_state and st.session_state["results"]:
        rl, rc, rr = st.columns([3, 2, 3])
        with rc:
            if st.button("↻ Reset & re-extract", use_container_width=True,
                          key="reset_btn"):
                st.cache_data.clear()
                st.session_state.pop("results", None)
                st.session_state.pop("file_signature", None)
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PROCESSING + 4. RESULTS
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

    # ── STAGE 4: MACRO FINANCIAL DASHBOARD (only when 2+ receipts) ──────
    successful = [r for r in results if "error" not in r]
    if len(successful) >= 2:
        clean_results = [{k: v for k, v in r.items() if not k.startswith("_")}
                         for r in results]
        summary = generate_summary(clean_results)
        sym0 = successful[0].get("currency_symbol", "$")

        st.markdown(f"""
        <div class="macro-banner">
          <div class="macro-banner-eyebrow">◇ Stage 4 · Aggregated Financial Summary</div>
          <div class="macro-banner-title">{summary['successful_extractions']} receipts processed</div>
          <div class="macro-banner-sub">Macro-level insights across all uploaded receipts.</div>
        </div>
        """, unsafe_allow_html=True)

        # Top KPI row
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">Total Spend</div>
              <div class="kpi-value"><span class="symbol">{sym0}</span>{summary['total_spend']:,.2f}</div>
              <div class="kpi-meta">{summary['successful_extractions']} transactions</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">Daily Average</div>
              <div class="kpi-value"><span class="symbol">{sym0}</span>{summary['daily_average']:,.2f}</div>
              <div class="kpi-meta">avg per active day</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">Top Category</div>
              <div class="kpi-value" style="font-size:1.4rem;font-family:Inter,sans-serif">{summary['top_category']}</div>
              <div class="kpi-meta">{len(summary['spend_per_category'])} categories detected</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            pct = int(summary['overall_confidence'] * 100)
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">Confidence</div>
              <div style="position:relative" class="donut-wrap">
                <div class="donut" style="--pct:{pct};position:relative">
                  <span>{pct}%</span>
                </div>
                <div class="donut-content">
                  <div style="color:var(--text-1);font-size:0.78rem;font-weight:600">
                    {summary['validated_fields_count']} validated
                  </div>
                  <div style="color:var(--text-3);font-size:0.7rem">
                    of {summary['extracted_fields_count']} extracted fields
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)

        # Per-store + categorized transactions
        c1, c2 = st.columns([1.1, 1], gap="large")
        with c1:
            st.markdown(section_head("Spend per Store",
                          f"{len(summary['spend_per_store'])} stores"),
                        unsafe_allow_html=True)
            if summary['spend_per_store']:
                store_df = pd.DataFrame([
                    {"Store": s[:25], "Total": v["total"]}
                    for s, v in list(summary['spend_per_store'].items())[:8]
                ])
                st.bar_chart(store_df.set_index("Store"),
                             color="#00c896", height=280)

        with c2:
            st.markdown(section_head("Spend by Category",
                          f"{len(summary['spend_per_category'])} categories"),
                        unsafe_allow_html=True)
            if summary['spend_per_category']:
                cat_df = pd.DataFrame([
                    {"Category": cat, "Total": v["total"]}
                    for cat, v in summary['spend_per_category'].items()
                ])
                st.bar_chart(cat_df.set_index("Category"),
                             color="#4ade80", height=280)

        # Categorized transactions table (matches blueprint page 2)
        st.markdown(section_head("Categorized Transactions"), unsafe_allow_html=True)
        rows_html = ""
        for r in successful[:15]:
            conf_pct = int(r.get("overall_conf", 0) * 100)
            status_class = "txn-status-ok" if conf_pct >= 70 else "txn-status-flag"
            status_text = "ACCEPT" if conf_pct >= 70 else "REVIEW"
            rows_html += (
                f'<tr>'
                f'<td>{r.get("date") or "—"}</td>'
                f'<td>{(r.get("store_name") or "—")[:25]}</td>'
                f'<td><span class="cat-pill">{r.get("category", "—")}</span></td>'
                f'<td style="text-align:right">{sym0} {r.get("total_amount") or "—"}</td>'
                f'<td style="text-align:right">{conf_pct}%</td>'
                f'<td style="text-align:right"><span class="{status_class}">{status_text}</span></td>'
                f'</tr>'
            )
        st.markdown(
            '<table class="txn-table">'
            '<thead><tr>'
            '<th>Date</th><th>Merchant</th><th>Category</th>'
            '<th style="text-align:right">Amount</th>'
            '<th style="text-align:right">Conf.</th>'
            '<th style="text-align:right">Status</th>'
            '</tr></thead>'
            f'<tbody>{rows_html}</tbody></table>',
            unsafe_allow_html=True
        )

        # Download summary JSON
        st.markdown('<div style="margin-top:1.2rem"></div>', unsafe_allow_html=True)
        dl_l, dl_c, dl_r = st.columns([2, 3, 2])
        with dl_c:
            st.download_button(
                "↓ Download Aggregated Summary (JSON)",
                data=json.dumps(summary, indent=2),
                file_name="financial_summary.json",
                mime="application/json",
                use_container_width=True,
                key="dl_summary",
            )

        st.markdown('<div style="margin-top:2.5rem"></div>', unsafe_allow_html=True)
        st.markdown(section_head("Per-Receipt Breakdown",
                                  f"{len(results)} receipts"),
                    unsafe_allow_html=True)

    for idx, r in enumerate(results):
        if "error" in r:
            st.error(f"❌ {r.get('filename', 'file')}: {r['error']}")
            continue

        st.markdown('<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
        col_img, col_data = st.columns([1, 1.15], gap="large")

        # LEFT — Image preview
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

        # RIGHT — Result card (money-first design)
        with col_data:
            opct = int(r["overall_conf"] * 100)
            sym  = r["currency_symbol"]
            code = r["currency_code"]
            total_str = r["total_amount"] or "—"
            subtotal_str = r["subtotal"] or "—"
            tax_str = r["tax"] or "—"

            # Build money breakdown rows (only show if data exists)
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

            # Carbon equivalence message — make the abstract number relatable
            carbon = r["carbon_kg"]
            if carbon > 0:
                # Rough analogy: 1 km of car driving ≈ 0.18 kg CO2e
                km_equiv = carbon / 0.18
                tree_days = carbon / 0.06   # 1 tree absorbs ~22kg/yr ≈ 0.06kg/day
                ctx = (
                    f'Equivalent to <strong>{km_equiv:.1f} km</strong> of car driving '
                    f'or <strong>{tree_days:.1f} days</strong> of CO₂ absorbed by one tree.'
                )
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

                  # ━━━ CO₂ HERO — Carbon Crunch's primary product ━━━
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

                  # ━━━ MONEY INPUT — primary financial value ━━━
                  '<div class="money-input">'
                    '<div class="money-input-label">'
                      '<span>◆ Total Amount Detected</span>'
                      f'<span class="currency-tag">{code} · {r.get("category","Uncategorized")}</span>'
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

            # JSON shape matches the Carbon Crunch blueprint (page 1):
            #   { "transaction": { date, total_amount, currency, items[],
            #                       tax, confidence_score }, "fields": {...} }
            payload = {
                "file": r["filename"],
                "transaction": {
                    "date":         r["date"],
                    "store_name":   r["store_name"],
                    "category":     r.get("category", "Uncategorized"),
                    "currency":     code,
                    "subtotal":     float(r["subtotal"]) if r.get("subtotal") else None,
                    "tax":          float(r["tax"])      if r.get("tax")      else None,
                    "total_amount": float(r["total_amount"]) if r.get("total_amount") else None,
                    "items": [
                        {
                            "description": it.get("description", it.get("name", "")),
                            "quantity":    it.get("quantity", 1),
                            "price":       float(it["price"]),
                        }
                        for it in r.get("items", [])
                    ],
                    "confidence_score": round(r["overall_conf"], 3),
                },
                "fields": {
                    "store_name":   {"value": r["store_name"],   "confidence": round(r["store_conf"], 3)},
                    "date":         {"value": r["date"],         "confidence": round(r["date_conf"], 3)},
                    "total_amount": {"value": r["total_amount"], "confidence": round(r["total_conf"], 3)},
                    "category":     {"value": r.get("category"), "confidence": round(r.get("category_conf", 0), 3)},
                },
                "low_confidence_flags": r.get("low_confidence_flags", []),
                "carbon_estimate_kg_co2e": r["carbon_kg"],
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
# 5. HOW IT WORKS
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
        Images are processed locally — nothing leaves your machine.
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
# 6. FOOTER
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
