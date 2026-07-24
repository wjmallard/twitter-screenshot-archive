"""Explore tools — summarize_period, list_topics, top_users, similar_users."""

import numpy as np

from ..core.db import get_conn
from .clustering import _cluster, _fetch_relevant
from .config import (
    SNIPPET_MAX_CHARS,
    SUMMARIZE_SNIPPETS,
)
from .server import mcp
from .utils import (
    add_time_filter,
    merge_similar_handles,
    normalize_handle,
)


def _pick_snippets(medoid: dict, members: list[dict], max_snippets: int) -> list[dict]:
    """Pick up to max_snippets members: most similar to medoid, sorted chronologically."""
    members = [m for m in members if m["id"] != medoid["id"]]
    if len(members) <= max_snippets:
        selected = members
    else:
        medoid_vec = medoid["embedding"]
        medoid_norm = np.linalg.norm(medoid_vec)
        if medoid_norm > 0:
            medoid_vec = medoid_vec / medoid_norm

        scored = []
        for m in members:
            vec = m["embedding"]
            norm = np.linalg.norm(vec)
            sim = float(np.dot(medoid_vec, vec / norm)) if norm > 0 else 0.0
            scored.append((sim, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [m for _, m in scored[:max_snippets]]

    selected.sort(key=lambda r: r["tweet_time"] or r["created_at"])
    return selected


@mcp.tool()
def summarize_period(
    after: str | None = None,
    before: str | None = None,
    topics: list[str] | None = None,
    users: list[str] | None = None,
    max_topics: int = 10,
) -> str:
    """Cluster and summarize what happened in a time window or around
    specific topics or users. Returns rich detail per topic: date span,
    top users, representative snippets. Use for overviews, discourse
    tracing, and narrative reconstruction.

    Args:
        after: Only include tweets after this date (YYYY-MM-DD).
        before: Only include tweets before this date (YYYY-MM-DD).
        topics: Optional topic strings to filter by (e.g. ["AI", "China trade"]).
        users: Optional list of handles to filter by (e.g. ["someone"]).
               Union semantics — includes any tweet mentioning any listed user.
        max_topics: Maximum number of topic clusters to return (default 10).
    """
    if not after and not before and not topics and not users:
        return "Error: provide at least one of after/before date range, topics, or users."

    rows = _fetch_relevant(after=after, before=before, topics=topics, users=users)
    if not rows:
        return "No tweets found in the specified range."

    clusters = _cluster(rows, max_topics=max_topics)
    if not clusters:
        return "No tweet clusters formed — too few tweets."

    parts = []
    for i, c in enumerate(clusters, 1):
        start_date = c["start_date"].strftime("%Y-%m-%d")
        end_date = c["end_date"].strftime("%Y-%m-%d")
        header = f"--- Topic {i} ({c['count']} tweets, {start_date} — {end_date}) ---"
        lines = [header]

        if c["top_users"]:
            lines.append("Top users: " + ", ".join("@" + u for u in c["top_users"]))

        medoid_snippet = (c["medoid"]["ocr_text_clean"] or "")[:SNIPPET_MAX_CHARS]
        lines.append(f"Representative: {medoid_snippet}")

        if SUMMARIZE_SNIPPETS > 0:
            snippet_rows = _pick_snippets(c["medoid"], c["members"], max_snippets=SUMMARIZE_SNIPPETS)
            lines.append("")
            for row in snippet_rows:
                t = (row["tweet_time"] or row["created_at"]).isoformat()
                snippet = (row["ocr_text_clean"] or "")[:SNIPPET_MAX_CHARS]
                lines.append(f"[ID {row['id']}] {t} | {snippet}")

        parts.append("\n".join(lines))

    return "\n\n".join(parts)


@mcp.tool()
def list_topics(
    after: str | None = None,
    before: str | None = None,
    users: list[str] | None = None,
    max_topics: int = 10,
) -> str:
    """List the main topics in a time window, ranked by tweet count.
    Lightweight overview — use summarize_period for detail.

    Args:
        after: Only include tweets after this date (YYYY-MM-DD).
        before: Only include tweets before this date (YYYY-MM-DD).
        users: Optional list of handles to filter by (e.g. ["someone"]).
               Union semantics — includes any tweet mentioning any listed user.
        max_topics: Maximum number of topics to return (default 10).
    """
    if not after and not before and not users:
        return "Error: provide at least one of after/before date range or users."

    rows = _fetch_relevant(after=after, before=before, users=users)
    if not rows:
        return "No tweets found in the specified range."

    clusters = _cluster(rows, max_topics=max_topics)
    if not clusters:
        return "No tweet clusters formed — too few tweets."

    lines = []
    for i, c in enumerate(clusters, 1):
        start_date = c["start_date"].strftime("%b %d")
        end_date = c["end_date"].strftime("%b %d")
        medoid_snippet = (c["medoid"]["ocr_text_clean"] or "")[:200]
        lines.append(f"{i}. {medoid_snippet} ({c['count']} tweets, {start_date}–{end_date})")

    return "\n".join(lines)


@mcp.tool()
def top_users(
    query: str | None = None,
    after: str | None = None,
    before: str | None = None,
    limit: int = 10,
) -> str:
    """Find which users appear most in tweets about a topic.

    Args:
        query: Topic to search for (e.g. "AI regulation"). If omitted, counts
               across all tweets in the date range.
        after: Only include tweets after this date (YYYY-MM-DD).
        before: Only include tweets before this date (YYYY-MM-DD).
        limit: Number of top users to return (default 10).
    """
    topics = [query] if query else None
    if not after and not before and not topics:
        return "Error: provide at least a query or a date range."

    rows = _fetch_relevant(after=after, before=before, topics=topics)
    if not rows:
        return "No tweets found."

    user_counts: dict[str, int] = {}
    for r in rows:
        for u in (r["mentioned_users"] or []):
            user_counts[u] = user_counts.get(u, 0) + 1

    user_counts = merge_similar_handles(user_counts)

    if not user_counts:
        return "No users mentioned in matching tweets."

    ranked = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:limit]

    label = f'about "{query}"' if query else "in range"
    header = f"Top users {label} ({len(rows)} tweets)"
    lines = [header]
    for i, (handle, count) in enumerate(ranked, 1):
        lines.append(f"{i}. @{handle} ({count})")

    return "\n".join(lines)


_SIMILAR_USERS_K = 5


@mcp.tool()
def similar_users(
    handle: str,
    after: str | None = None,
    before: str | None = None,
    limit: int = 10,
    k: int = _SIMILAR_USERS_K,
) -> str:
    """Find users who appear in tweets about similar topics to a given user.

    For each tweet mentioning the target handle, finds the K nearest tweets
    (by embedding) that don't mention that handle, then aggregates users
    from those neighbors. Handles bimodal users naturally — each tweet
    finds its own neighbors independently.

    Args:
        handle: Twitter handle to find similar users for (with or without @).
        after: Only include tweets after this date (YYYY-MM-DD).
        before: Only include tweets before this date (YYYY-MM-DD).
        limit: Number of similar users to return (default 10).
        k: Neighbors per source tweet (default 5).
    """
    handle = normalize_handle(handle)

    # Fetch embeddings for all tweets mentioning this user
    source_conditions = [
        "%(handle)s = ANY(mentioned_users)",
        "embedding IS NOT NULL",
    ]
    params: dict = {"handle": handle}
    add_time_filter(source_conditions, params, after, before)
    source_where = " AND ".join(source_conditions)

    with get_conn() as conn:
        source_rows = conn.execute(
            f"""
            SELECT
                id,
                embedding::text
            FROM screenshots
            WHERE {source_where}
            """,
            params,
        ).fetchall()

    if not source_rows:
        return f"No embedded tweets found mentioning @{handle}"

    # For each source tweet, find K nearest neighbors that don't mention the handle
    neighbor_conditions = [
        "embedding IS NOT NULL",
        "NOT (%(handle)s = ANY(mentioned_users))",
    ]
    neighbor_params: dict = {
        "handle": handle,
        "k": k,
    }
    add_time_filter(neighbor_conditions, neighbor_params, after, before)
    neighbor_where = " AND ".join(neighbor_conditions)

    user_counts: dict[str, int] = {}
    neighbor_count = 0

    with get_conn() as conn:
        for source in source_rows:
            # embedding::text is already a pgvector literal — pass it back as-is
            neighbor_params["vec"] = source["embedding"]
            neighbors = conn.execute(
                f"""
                SELECT mentioned_users
                FROM screenshots
                WHERE {neighbor_where}
                ORDER BY embedding <=> %(vec)s::vector
                LIMIT %(k)s
                """,
                neighbor_params,
            ).fetchall()

            for neighbor in neighbors:
                neighbor_count += 1
                for u in (neighbor["mentioned_users"] or []):
                    user_counts[u] = user_counts.get(u, 0) + 1

    user_counts = merge_similar_handles(user_counts, primary=handle)

    if not user_counts:
        return f"No other users found in tweets similar to @{handle}"

    ranked = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:limit]

    header = f"Users similar to @{handle} ({len(source_rows)} source tweets, {neighbor_count} neighbors)"
    lines = [header]
    for i, (u, count) in enumerate(ranked, 1):
        lines.append(f"{i}. @{u} ({count})")

    return "\n".join(lines)
