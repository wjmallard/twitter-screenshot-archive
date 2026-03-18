"""Standalone VLM description backfill entry point (tsa-describe)."""

from .vision import backfill_descriptions, load_vlm


def main():
    from ..core.db import check_db
    check_db()
    load_vlm()
    backfill_descriptions()
