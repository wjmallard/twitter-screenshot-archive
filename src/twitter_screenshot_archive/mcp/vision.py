"""VLM image description engine and backfill (tsa-describe)."""

import logging
import re
import sys

logging.getLogger("httpx").setLevel(logging.WARNING)

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from tqdm import tqdm

from ..core.db import get_conn
from .config import VLM_MAX_TOKENS, VLM_MODEL_ID, VLM_REPETITION_PENALTY

_PROMPT = """\
Parse this image. If it contains tweets, transcribe each tweet top to bottom as:
@handle [timestamp]: "verbatim tweet text"

If a tweet quotes another tweet, add it indented on the next line:
  Quoting @handle [timestamp]: "verbatim quoted text"

If a tweet contains an embedded image, describe it briefly in parentheses.

Use exact timestamps if visible (3:05 PM · 7/11/24) or relative ones (3h, 2d).
Do NOT include likes, retweets, views, bookmarks, emoji, or any other metadata.
Output only the transcribed tweets, nothing else.

If this is not a tweet screenshot, just describe what the image shows in one or two sentences."""

_model = None
_processor = None


def load_vlm():
    """Load the VLM and processor into module state."""
    global _model, _processor
    print(f"Loading VLM {VLM_MODEL_ID}...", file=sys.stderr)
    _model, _processor = load(VLM_MODEL_ID)
    print("VLM ready.", file=sys.stderr)


def _parse_type(text: str) -> str:
    """Infer image type from VLM output structure."""
    lower = text.lower()
    # @handle followed by timestamp-like content and a colon (with or without quotes)
    if re.search(r"@\w+.*?:", lower):
        return "twitter_screenshot"
    if re.search(r"quoting\s+@\w+", lower):
        return "twitter_screenshot"
    return "downloaded_content"


def _parse_description(text: str) -> str:
    """Clean up VLM output into the stored description."""
    return text.strip()


def describe_image(image_path: str) -> tuple[str, str]:
    """Run the VLM on a single image.

    Returns (image_type, image_description).
    """
    prompt = apply_chat_template(
        _processor, _model.config, _PROMPT, num_images=1,
    )
    result = generate(
        _model,
        _processor,
        prompt,
        image=[image_path],
        max_tokens=VLM_MAX_TOKENS,
        temperature=0.0,
        repetition_penalty=VLM_REPETITION_PENALTY,
        prefill_step_size=None,
    )
    raw = result.text
    return _parse_type(raw), _parse_description(raw)


def backfill_descriptions():
    """Describe all rows that have no image_description yet."""
    with get_conn() as conn:
        pending = conn.execute(
            "SELECT count(*) FROM screenshots WHERE image_description IS NULL"
        ).fetchone()["count"]

        if pending == 0:
            print("All images already described.", file=sys.stderr)
            return

        progress = tqdm(total=pending, desc="Describing images", file=sys.stderr)

        while True:
            rows = conn.execute(
                "SELECT id, file_path FROM screenshots "
                "WHERE image_description IS NULL "
                "ORDER BY id LIMIT 100",
            ).fetchall()

            if not rows:
                break

            for row in rows:
                try:
                    image_type, description = describe_image(row["file_path"])
                except Exception as exc:
                    print(
                        f"\nWarning: Failed on id {row['id']}: {exc}",
                        file=sys.stderr,
                    )
                    image_type = None
                    description = f"[error: {exc}]"

                conn.execute(
                    "UPDATE screenshots "
                    "SET image_type = %(image_type)s, "
                    "    image_description = %(description)s "
                    "WHERE id = %(id)s",
                    {
                        "image_type": image_type,
                        "description": description,
                        "id": row["id"],
                    },
                )
                conn.commit()
                progress.update(1)

        progress.close()
