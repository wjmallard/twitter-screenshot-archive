"""Standalone embedding backfill entry point (tsa-embed)."""

from .embedding import backfill_embeddings, load_model


def main():
    from ..core.db import check_db
    check_db()
    load_model()
    backfill_embeddings()
