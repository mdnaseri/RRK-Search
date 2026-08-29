from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


COOKIE = os.getenv("RRK_COOKIE", "").strip() or None
START_DATE = os.getenv("START_DATE", "1360-01-01").strip()
END_DATE = os.getenv("END_DATE", "1360-12-29").strip()
SEARCH_PHRASES = tuple(
    p.strip()
    for p in os.getenv("SEARCH_PHRASES", "").split("|")
    if p.strip()
)

OCR_ENGINE = os.getenv("OCR_ENGINE", "paddleocr").strip().lower()
OCR_USE_GPU = _bool("OCR_USE_GPU", False)
OCR_PADDLE_LANG = os.getenv("OCR_PADDLE_LANG", "fa").strip()
OCR_TESSERACT_LANG = os.getenv("OCR_TESSERACT_LANG", "fas+ara").strip()

IMAGES_DIR = os.getenv("IMAGES_DIR", "images").strip()
RESULT_DIR = os.getenv("RESULT_DIR", "result").strip()
DOWNLOAD_WORKERS = _int("DOWNLOAD_WORKERS", 6)
VERIFY_SSL = _bool("VERIFY_SSL", False)
# How many threads process images on the GPU (only when OCR_USE_GPU=true).
# 0 = auto: 1 for paddleocr/easyocr, 0 for tesseract (no GPU support).
OCR_GPU_WORKERS = _int("OCR_GPU_WORKERS", 0)
# How many threads process images on the CPU in parallel with GPU workers.
# 0 = auto: cpu_count//2 (capped at 4) for tesseract, 2 for others.
OCR_CPU_WORKERS = _int("OCR_CPU_WORKERS", 0)
# Legacy alias: if only OCR_WORKERS is set it becomes the CPU worker count.
OCR_WORKERS = _int("OCR_WORKERS", 0)
