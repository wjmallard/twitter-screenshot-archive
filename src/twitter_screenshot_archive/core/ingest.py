"""Batch OCR runner for screenshot ingestion."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pillow_heif import register_heif_opener
from tqdm import tqdm

from . import config
from .cleaning import clean_ocr_text
from .dates import extract_tweet_time, parse_tz_offset
from .db import check_db, get_conn, images_in_db, upsert_screenshot
from .minhash import compute_signature
from .usernames import extract_usernames

register_heif_opener()

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".heic", ".tiff", ".bmp"}


def _dates_from_strings(dt_str, offset_str):
    """Convert EXIF-style date/offset strings to (utc, local_naive, tz_offset_str)."""
    if not dt_str:
        return None, None, None
    local_naive = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
    if not offset_str:
        return None, local_naive, None
    tz = parse_tz_offset(offset_str)
    utc = local_naive.replace(tzinfo=tz).astimezone(timezone.utc)
    return utc, local_naive, offset_str


def parse_dates_from_sidecar(image_path: Path) -> tuple[datetime | None, datetime | None, str | None]:
    """Parse dates from JSON sidecar. Returns (utc, local_naive, tz_offset_str)."""
    sidecar = Path(str(image_path) + ".json")
    if not sidecar.is_file():
        return None, None, None
    try:
        data = json.loads(sidecar.read_text())
        if isinstance(data, list):
            data = data[0]
        return _dates_from_strings(
            data.get("EXIF:DateTimeOriginal"),
            data.get("EXIF:OffsetTimeOriginal"),
        )
    except Exception:
        return None, None, None


def parse_dates_from_exif(img: Image.Image) -> tuple[datetime | None, datetime | None, str | None]:
    """Fallback: parse dates from EXIF data via Pillow. Returns (utc, local_naive, tz_offset_str)."""
    try:
        exif = img.getexif()
        if not exif:
            return None, None, None
        # 36867 = DateTimeOriginal, 36881 = OffsetTimeOriginal
        return _dates_from_strings(exif.get(36867), exif.get(36881))
    except Exception:
        return None, None, None


def preprocess(img: Image.Image) -> Image.Image:
    """Apply CLAHE contrast enhancement for dark-mode screenshot OCR."""
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    arr = np.array(img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(16, 16))
    return Image.fromarray(clahe.apply(arr))


def process_image(path: Path) -> dict:
    """Run OCR and extract metadata for a single image. Returns a dict for upsert."""
    img = Image.open(path)
    width, height = img.size
    ocr_text = pytesseract.image_to_string(preprocess(img))
    # Try sidecar first, fall back to EXIF
    utc, local, tz = parse_dates_from_sidecar(path)
    if local is None:
        utc, local, tz = parse_dates_from_exif(img)
    tweet_time, tweet_time_source = extract_tweet_time(ocr_text, utc, tz)
    return {
        "file_path": str(path),
        "ocr_text": ocr_text,
        "ocr_text_clean": clean_ocr_text(ocr_text),
        "created_at": utc,
        "created_at_local": local,
        "timezone": tz,
        "width": width,
        "height": height,
        "file_size": path.stat().st_size,
        "minhash_signature": compute_signature(ocr_text),
        "mentioned_users": extract_usernames(ocr_text),
        "tweet_time": tweet_time,
        "tweet_time_source": tweet_time_source,
    }


def images_on_disk(root: Path) -> set[str]:
    """Recursively find image files under root, returned as absolute path strings."""
    return {
        str(p) for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    }


def ingest(root: Path, workers: int = config.TESSERACT_WORKERS):
    on_disk = images_on_disk(root)
    print(f"Found {len(on_disk)} images under {root}")

    with get_conn() as conn:
        in_db = images_in_db(conn)

    to_process = sorted(on_disk - in_db)
    skipped = len(on_disk) - len(to_process)
    if skipped:
        print(f"Skipping {skipped} already-ingested files")
    if not to_process:
        print("Nothing to ingest.")
        return

    errors = 0
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_image, Path(p)): p for p in to_process}
        with get_conn() as conn:
            for future in tqdm(as_completed(futures), total=len(to_process), unit="img"):
                path = futures[future]
                try:
                    upsert_screenshot(conn, future.result())
                    done += 1
                    if done % config.COMMIT_BATCH_SIZE == 0:
                        conn.commit()
                except Exception as e:
                    errors += 1
                    tqdm.write(f"ERROR {path}: {e}", file=sys.stderr)

    print(f"\nDone ({errors} errors)" if errors else "\nDone")


def main():
    check_db()
    ingest(config.SCREENSHOT_DIR)


if __name__ == "__main__":
    main()
