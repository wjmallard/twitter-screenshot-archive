"""Standalone VLM description backfill entry point (tsa-describe)."""

import sys

from ..core.db import check_db


def main():
    check_db()
    try:
        from .vision import backfill_descriptions, load_vlm
    except ImportError:
        print(
            "Error: MLX dependencies are not installed.\n"
            "Install the mcp extra:  uv sync --extra mcp",
            file=sys.stderr,
        )
        sys.exit(1)
    load_vlm()
    backfill_descriptions()
