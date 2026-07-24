"""In-process MLX embedding engine and startup backfill."""

import sys

import mlx.core as mx
from mlx_lm import load as mlx_load
from tqdm import tqdm

from ..core.db import get_conn
from .config import BACKFILL_BATCH_SIZE, EMBEDDING_MODEL_ID

_model = None
_tokenizer = None


def load_model():
    """Load the embedding model and tokenizer into module state."""
    global _model, _tokenizer
    print(f"Loading embedding model {EMBEDDING_MODEL_ID}...", file=sys.stderr)
    _model, _tokenizer = mlx_load(EMBEDDING_MODEL_ID)
    mx.eval(_model.parameters())
    print("Embedding model ready.", file=sys.stderr)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using the loaded MLX model.

    Returns a list of float lists (one embedding per input text).
    Uses last-token pooling with L2 normalization (Qwen3-Embedding convention).
    """
    tokens = _tokenizer._tokenizer(
        texts,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=512,
    )
    input_ids = mx.array(tokens["input_ids"])
    attention_mask = mx.array(tokens["attention_mask"])

    # Forward pass through transformer body (skip LM head)
    hidden = _model.model(input_ids)

    # Last-token pooling: extract hidden state at last non-pad position
    seq_lengths = attention_mask.sum(axis=1) - 1
    batch_idx = mx.arange(hidden.shape[0])
    embeds = hidden[batch_idx, seq_lengths]

    # L2 normalize
    norms = mx.linalg.norm(embeds, axis=1, keepdims=True)
    embeds = embeds / mx.where(norms == 0, 1, norms)

    mx.eval(embeds)
    return embeds.tolist()


def vec_literal(embedding: list[float]) -> str:
    """Format a float list as a pgvector text literal, e.g. '[0.1,0.2,...]'."""
    return "[" + ",".join(str(v) for v in embedding) + "]"


def backfill_embeddings():
    """Embed all rows that have ocr_text_clean but no embedding yet."""
    with get_conn() as conn:
        pending = conn.execute(
            """
            SELECT count(*)
            FROM screenshots
            WHERE ocr_text_clean IS NOT NULL
              AND ocr_text_clean != ''
              AND embedding IS NULL
            """
        ).fetchone()["count"]

        if pending == 0:
            return

        progress = tqdm(total=pending, desc="Embedding backfill", file=sys.stderr)

        while True:
            rows = conn.execute(
                """
                SELECT
                    id,
                    ocr_text_clean
                FROM screenshots
                WHERE ocr_text_clean IS NOT NULL
                  AND ocr_text_clean != ''
                  AND embedding IS NULL
                ORDER BY id
                LIMIT %(limit)s
                """,
                {
                    "limit": BACKFILL_BATCH_SIZE,
                },
            ).fetchall()

            if not rows:
                break

            # Sort by text length to minimize padding waste within the batch
            pairs = sorted(rows, key=lambda r: len(r["ocr_text_clean"]))
            ids = [r["id"] for r in pairs]
            texts = [r["ocr_text_clean"] for r in pairs]
            embeddings = embed_texts(texts)

            for row_id, emb in zip(ids, embeddings):
                conn.execute(
                    "UPDATE screenshots SET embedding = %(vec)s::vector WHERE id = %(id)s",
                    {
                        "vec": vec_literal(emb),
                        "id": row_id,
                    },
                )
            conn.commit()
            progress.update(len(ids))

        progress.close()
