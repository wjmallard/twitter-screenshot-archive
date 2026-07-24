"""search_tweets tool."""

from ..core.db import get_conn
from .config import (
    DEFAULT_SEARCH_LIMIT,
    SEARCH_SIMILARITY_FLOOR,
    SNIPPET_MAX_CHARS,
)
from .embedding import embed_texts, vec_literal
from .server import mcp
from .utils import (
    add_time_filter,
    add_users_filter,
    validate_keywords,
)


def _format_result(index: int, total: int, row) -> str:
    """Format a single search result as plain text."""
    lines = [f"[{index}/{total}] ID {row['id']}"]
    sim = row.get("similarity")
    if sim is not None:
        lines[0] += f" | sim: {sim:.2f}"
    if row["tweet_time"]:
        lines[0] += f" | {row['tweet_time'].isoformat()}"
    if row["mentioned_users"]:
        lines.append(f"Users: {', '.join('@' + u for u in row['mentioned_users'])}")
    snippet = (row["ocr_text_clean"] or "")[:SNIPPET_MAX_CHARS]
    if snippet:
        lines.append(snippet)
    return "\n".join(lines)


@mcp.tool()
def search_tweets(
    query: str | None = None,
    keywords: str | None = None,
    limit: int = DEFAULT_SEARCH_LIMIT,
    offset: int = 0,
    min_score: float = SEARCH_SIMILARITY_FLOOR,
    after: str | None = None,
    before: str | None = None,
    users: list[str] | None = None,
    sort: str = "relevance",
) -> str:
    """Find tweets by meaning and/or keywords. Semantic search matches meaning;
    keywords filter by exact words. Use both together for precision.

    Args:
        query: Semantic search query (matches meaning, not exact words).
               Required unless keywords is provided.
        keywords: PostgreSQL tsquery filter for exact word matching.
                  Use '|' for OR, '&' for AND, '!' for NOT.
                  Examples: "dementia | senile", "biden & !genocide"
        limit: Max results to return (default 10).
        offset: Number of results to skip (default 0). Use to paginate.
        min_score: Minimum similarity threshold (default 0.4). Only applies
                   when query is provided. Set to 0 to disable.
        after: Only include tweets after this date (YYYY-MM-DD).
        before: Only include tweets before this date (YYYY-MM-DD).
        users: Optional list of handles to filter by (e.g. ["someone"]).
               Union semantics — includes any tweet mentioning any listed user.
        sort: "relevance" (default) or "chronological" (newest first).
    """
    if not query and not keywords:
        return "Error: provide at least one of query or keywords."

    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    conditions = []
    params: dict = {
        "limit": limit,
        "offset": offset,
    }

    has_query = bool(query)

    if query:
        query_emb = embed_texts([query])[0]
        vec = vec_literal(query_emb)
        conditions.append("embedding IS NOT NULL")
        conditions.append("1 - (embedding <=> %(vec)s::vector) >= %(floor)s")
        params["vec"] = vec
        params["floor"] = min_score

    if keywords:
        if not validate_keywords(keywords):
            return f"Error: invalid keywords syntax: {keywords!r}"
        conditions.append("ocr_text_tsv @@ to_tsquery('english', %(keywords)s)")
        params["keywords"] = keywords

    add_time_filter(conditions, params, after, before)
    add_users_filter(conditions, params, users)

    where = " AND ".join(conditions) if conditions else "TRUE"

    # Determine SELECT and ORDER BY based on mode
    if has_query:
        select_extra = ", 1 - (embedding <=> %(vec)s::vector) AS similarity"
        if sort == "chronological":
            order_by = "COALESCE(tweet_time, created_at) DESC"
        else:
            order_by = "embedding <=> %(vec)s::vector"
    else:
        select_extra = ", ts_rank(ocr_text_tsv, to_tsquery('english', %(keywords)s)) AS rank"
        if sort == "chronological":
            order_by = "COALESCE(tweet_time, created_at) DESC"
        else:
            order_by = "ts_rank(ocr_text_tsv, to_tsquery('english', %(keywords)s)) DESC"

    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT
                id,
                ocr_text_clean,
                tweet_time,
                mentioned_users,
                count(*) OVER () AS total_count
                {select_extra}
            FROM screenshots
            WHERE {where}
            ORDER BY {order_by}
            LIMIT %(limit)s
            OFFSET %(offset)s
            """,
            params,
        ).fetchall()

    if not rows:
        return "No results found."

    total = rows[0]["total_count"]
    sort_label = "newest first" if sort == "chronological" else "best match first"
    start = offset + 1
    end = offset + len(rows)

    # Build header
    desc_parts = []
    if query:
        desc_parts.append(query)
    if keywords:
        desc_parts.append(f"keywords={keywords!r}")
    desc = ", ".join(desc_parts)

    if len(rows) < total - offset:
        header = f"Results {start}\u2013{end} of {total} for: {desc} ({sort_label})"
    else:
        header = f"Found {total} results for: {desc} ({sort_label})"
    parts = [header + "\n"]
    for i, row in enumerate(rows, start):
        parts.append(_format_result(i, total, row))

    return "\n\n".join(parts)
