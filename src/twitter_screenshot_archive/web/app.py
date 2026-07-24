"""Flask search GUI for screenshot search."""

import argparse
import hashlib
import mimetypes
import os
import tempfile
from importlib.util import find_spec
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

from ..core import config, embedding
from ..core.db import (
    check_db, get_conn, search_fulltext, search_trigram, search_exact,
    search_semantic, count_fulltext, count_trigram, count_exact,
    count_semantic, count_screenshots,
    get_screenshots_by_ids, get_timeline_neighbors,
)
from ..core.minhash import load_or_build_lsh_index, query_related

app = Flask(__name__)

PER_PAGE = config.RESULTS_PER_PAGE

_IMAGE_ROOT = config.SCREENSHOT_DIR.resolve()

_THUMB_DIR = Path.home() / ".cache" / "twitter-screenshot-archive" / "thumbs"

# Semantic search needs MLX (installed with the mcp extra); hide the
# option when it isn't available rather than erroring.
SEMANTIC_AVAILABLE = find_spec("mlx_lm") is not None
SEMANTIC_MIN_SCORE = config.RAW.get("search_similarity_floor", 0.4)

_lsh = None
_minhashes = {}


def _init_index():
    """Build or load the LSH index once."""
    global _lsh, _minhashes
    if _lsh is None:
        _lsh, _minhashes = load_or_build_lsh_index()


def _format_size(size_bytes):
    """Format file size in human-readable units."""
    if size_bytes is None:
        return ""
    for unit in ("B", "kB", "MB", "GB"):
        if size_bytes < 1000 or unit == "GB":
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} B"
        size_bytes /= 1000


def _page_numbers(current, total):
    """Generate page numbers with ellipsis. Always returns exactly 7 slots when total >= 7."""
    if total <= 7:
        return list(range(1, total + 1))
    # Near the start: 1 2 3 4 5 ... last
    if current <= 4:
        return [1, 2, 3, 4, 5, None, total]
    # Near the end: 1 ... n-4 n-3 n-2 n-1 n
    if current >= total - 3:
        return [1, None, total - 4, total - 3, total - 2, total - 1, total]
    # Middle: 1 ... c-1 c c+1 ... last
    return [1, None, current - 1, current, current + 1, None, total]


@app.route("/")
def index():
    q = request.args.get("q", "").strip()
    fuzzy = request.args.get("fuzzy", "word")
    sort = request.args.get("sort", "best")
    after = request.args.get("after", "").strip()
    before = request.args.get("before", "").strip()
    page = request.args.get("page", 1, type=int)
    offset = (page - 1) * PER_PAGE
    results = []
    total_results = 0

    if q:
        with get_conn() as conn:
            if fuzzy == "char":
                rows = search_trigram(conn, q, limit=PER_PAGE, offset=offset, sort=sort, after=after, before=before)
                total_results = count_trigram(conn, q, after=after, before=before)
            elif fuzzy == "none":
                rows = search_exact(conn, q, limit=PER_PAGE, offset=offset, sort=sort, after=after, before=before)
                total_results = count_exact(conn, q, after=after, before=before)
            elif fuzzy == "semantic" and SEMANTIC_AVAILABLE:
                embedding.ensure_model()
                query_vec = embedding.vec_literal(embedding.embed_texts([q])[0])
                rows = search_semantic(conn, query_vec, limit=PER_PAGE, offset=offset, sort=sort, min_score=SEMANTIC_MIN_SCORE, after=after, before=before)
                total_results = count_semantic(conn, query_vec, min_score=SEMANTIC_MIN_SCORE, after=after, before=before)
            else:
                rows = search_fulltext(conn, q, limit=PER_PAGE, offset=offset, sort=sort, after=after, before=before)
                total_results = count_fulltext(conn, q, after=after, before=before)
            results.extend(_format_screenshot(row) for row in rows)

    total_pages = (total_results + PER_PAGE - 1) // PER_PAGE if total_results else 0

    with get_conn() as conn:
        total_indexed = count_screenshots(conn)

    return render_template(
        "index.html",
        q=q,
        fuzzy=fuzzy,
        sort=sort,
        after=after,
        before=before,
        page=page,
        total_pages=total_pages,
        results=results,
        total_indexed=total_indexed,
        total_results=total_results,
        pages=_page_numbers(page, total_pages),
        semantic_available=SEMANTIC_AVAILABLE,
    )


@app.route("/image")
def serve_image():
    path = request.args.get("path", "")
    if not path:
        abort(400)
    p = Path(path).resolve()
    if not p.is_relative_to(_IMAGE_ROOT):
        abort(403)
    if not p.is_file():
        abort(404)

    if request.args.get("thumb"):
        fmt = "PNG" if p.suffix.lower() == ".png" else "JPEG"
        ext = ".png" if fmt == "PNG" else ".jpg"
        key = hashlib.sha1(f"{p}:{p.stat().st_mtime_ns}".encode()).hexdigest()
        cached = _THUMB_DIR / f"{key}{ext}"
        if not cached.is_file():
            img = Image.open(p)
            img.thumbnail((800, 800))
            if fmt == "JPEG" and img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            _THUMB_DIR.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=_THUMB_DIR, suffix=ext)
            with os.fdopen(fd, "wb") as f:
                img.save(f, format=fmt)
            os.replace(tmp, cached)
        mime = "image/png" if fmt == "PNG" else "image/jpeg"
        return send_file(cached, mimetype=mime, max_age=86400)

    mime = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
    return send_file(p, mimetype=mime, max_age=86400)


def _format_screenshot(row):
    """Shape a screenshots row for templates and JSON responses."""
    formatted = {
        "id": row["id"],
        "file_path": row["file_path"],
        "name": Path(row["file_path"]).name,
        "ocr_text": row["ocr_text"] or "",
        "date": row["created_at_local"].strftime("%Y-%m-%d · %I:%M %p · %A") if row["created_at_local"] else "unknown",
        "timezone": row["timezone"] or "",
        "width": row["width"],
        "height": row["height"],
        "file_size": _format_size(row["file_size"]),
    }
    if "score" in row:
        formatted["score"] = row["score"]
    return formatted


@app.route("/related/<int:screenshot_id>")
def related(screenshot_id):
    matches = query_related(_lsh, _minhashes, screenshot_id)
    match_ids = [mid for mid, _ in matches]
    sim_by_id = dict(matches)
    all_ids = [screenshot_id] + match_ids
    with get_conn() as conn:
        screenshots = get_screenshots_by_ids(conn, all_ids)
    source = None
    if screenshot_id in screenshots:
        source = _format_screenshot(screenshots[screenshot_id])
    related_results = []
    for mid in match_ids:
        if mid not in screenshots:
            continue
        r = _format_screenshot(screenshots[mid])
        r["similarity"] = round(sim_by_id[mid], 3)
        related_results.append(r)
    return jsonify({"source": source, "related": related_results})


@app.route("/timeline/<int:screenshot_id>")
def timeline(screenshot_id):
    with get_conn() as conn:
        before, focal, after = get_timeline_neighbors(conn, screenshot_id)
    if focal is None:
        abort(404)

    return jsonify({
        "before": [_format_screenshot(r) for r in before],
        "focal": _format_screenshot(focal),
        "after": [_format_screenshot(r) for r in after],
    })


def main():
    check_db()

    parser = argparse.ArgumentParser(description="Twitter Archive web UI")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode and auto-reload",
    )
    args = parser.parse_args()

    # With the debug reloader active, the parent process only monitors files
    # and never serves requests, so the index build belongs in the serving
    # child (which has WERKZEUG_RUN_MAIN set). Without the reloader there is
    # only one process — build directly.
    if not args.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        _init_index()
    app.run(debug=args.debug, port=config.FLASK_PORT)


if __name__ == "__main__":
    main()
