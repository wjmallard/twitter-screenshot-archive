"""Standalone embedding backfill entry point (tsa-embed)."""

import sys

from ..core.db import check_db


def main():
    check_db()
    try:
        from ..core.embedding import backfill_embeddings, load_model
    except ImportError:
        print(
            "Error: MLX dependencies are not installed.\n"
            "Install the mcp extra:  uv sync --extra mcp",
            file=sys.stderr,
        )
        sys.exit(1)
    load_model()
    backfill_embeddings()
