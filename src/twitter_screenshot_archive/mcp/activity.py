"""tweet_activity tool — histogram of tweet counts over time."""

from ..core.db import get_conn
from .config import SEARCH_SIMILARITY_FLOOR
from .embedding import embed_texts, vec_literal
from .server import mcp
from .utils import (
    add_time_filter,
    add_users_filter,
    validate_keywords,
)

_GRANULARITY_LABELS = {
    "day": "daily",
    "month": "monthly",
    "week": "weekly",
    "year": "yearly",
}


@mcp.tool()
def tweet_activity(
    query: str | None = None,
    keywords: str | None = None,
    users: list[str] | None = None,
    after: str | None = None,
    before: str | None = None,
    granularity: str = "year",
    min_score: float = SEARCH_SIMILARITY_FLOOR,
    include_mean: bool = False,
) -> str:
    """Show tweet counts over time, bucketed by day/week/month/year.
    Use to find when a topic or user was most active before drilling in.

    Args:
        query: Optional semantic search query to filter by topic.
        keywords: Optional PostgreSQL tsquery filter for exact word matching.
                  Use '|' for OR, '&' for AND, '!' for NOT.
        users: Optional list of handles to filter by (e.g. ["someone"]).
        after: Only include tweets after this date (YYYY-MM-DD).
        before: Only include tweets before this date (YYYY-MM-DD).
        granularity: Bucket size — "day", "week", "month", or "year" (default).
        min_score: Minimum similarity threshold when query is provided (default 0.4).
        include_mean: When query is provided, also show mean similarity per
                      bucket alongside the always-on max (default false).
    """
    if granularity not in _GRANULARITY_LABELS:
        return f"Error: granularity must be one of: {', '.join(sorted(_GRANULARITY_LABELS))}"

    if keywords and not validate_keywords(keywords):
        return f"Error: invalid keywords syntax: {keywords!r}"

    conditions = ["ocr_text_clean IS NOT NULL"]
    params: dict = {}

    if query:
        query_emb = embed_texts([query])[0]
        vec = vec_literal(query_emb)
        conditions.append("embedding IS NOT NULL")
        conditions.append(
            "1 - (embedding <=> %(vec)s::vector) >= %(floor)s"
        )
        params["vec"] = vec
        params["floor"] = min_score

    if keywords:
        conditions.append("ocr_text_tsv @@ to_tsquery('english', %(keywords)s)")
        params["keywords"] = keywords

    add_time_filter(conditions, params, after, before)
    add_users_filter(conditions, params, users)

    where = " AND ".join(conditions)

    # Build SELECT columns
    bucket = f"date_trunc(%(granularity)s, COALESCE(tweet_time, created_at))"
    params["granularity"] = granularity

    select_cols = [f"{bucket} AS bucket", "count(*) AS tweet_count"]
    if query:
        select_cols.append(
            f"MAX(1 - (embedding <=> %(vec)s::vector)) AS max_sim"
        )
        if include_mean:
            select_cols.append(
                f"AVG(1 - (embedding <=> %(vec)s::vector)) AS mean_sim"
            )

    select = ", ".join(select_cols)

    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT {select}
            FROM screenshots
            WHERE {where}
            GROUP BY bucket
            ORDER BY bucket
            """,
            params,
        ).fetchall()

    if not rows:
        parts = []
        if query:
            parts.append(f"query={query!r}")
        if users:
            parts.append(f"users={users}")
        return f"No activity found ({', '.join(parts) or 'no filters'})."

    # Find max count for bar scaling
    max_count = max(r["tweet_count"] for r in rows)
    total = sum(r["tweet_count"] for r in rows)
    bar_width = 20

    # Format output
    lines = []

    # Header
    header_parts = [f"{total} tweets"]
    if query:
        header_parts.append(f"matching {query!r}")
    if users:
        header_parts.append(f"by {', '.join('@' + u for u in users)}")
    header_parts.append(f"({_GRANULARITY_LABELS[granularity]})")
    lines.append(" ".join(header_parts))
    lines.append("")

    for row in rows:
        bucket_date = row["bucket"]
        count = row["tweet_count"]

        if granularity == "day":
            label = bucket_date.strftime("%Y-%m-%d")
        elif granularity == "week":
            label = f"{bucket_date.strftime('%Y-%m-%d')}w"
        elif granularity == "month":
            label = bucket_date.strftime("%Y-%m")
        else:
            label = bucket_date.strftime("%Y")

        bar_len = round(count / max_count * bar_width) if max_count > 0 else 0
        bar = "\u2588" * bar_len

        if query:
            score_str = f"max={row['max_sim']:.2f}"
            if include_mean:
                score_str += f"  mean={row['mean_sim']:.2f}"
            lines.append(f"{label}  {count:>5}  {score_str}  {bar}")
        else:
            lines.append(f"{label}  {count:>5}  {bar}")

    return "\n".join(lines)
