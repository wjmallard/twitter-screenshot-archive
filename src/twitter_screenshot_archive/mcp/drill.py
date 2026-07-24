"""Drill tools — get_tweet, nearby_screenshots, find_related, search_by_user, interactions."""

import html

from ..core.db import get_conn, get_timeline_neighbors
from ..core.minhash import query_related
from . import server
from .config import (
    SNIPPET_MAX_CHARS,
)
from .utils import (
    add_time_filter,
    dedup_handles,
    normalize_handle,
)

mcp = server.mcp


@mcp.tool()
def get_tweet(id: int) -> str:
    """Get the full OCR text of a specific tweet by ID, plus the exact
    text of any matching tweet from the owner's Twitter export."""
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT
                id,
                ocr_text_clean,
                tweet_time,
                mentioned_users,
                image_type,
                image_description
            FROM screenshots
            WHERE id = %(id)s
            """,
            {
                "id": id,
            },
        ).fetchone()

        if not row:
            return f"No screenshot with id {id}"

        matches = conn.execute(
            """
            SELECT
                m.tweet_id,
                m.score,
                COALESCE(t.full_text, l.full_text) AS full_text,
                COALESCE(t.created_at, l.snowflake_date) AS tweet_time,
                COALESCE(t.account, l.account) AS account,
                (t.tweet_id IS NOT NULL) AS authored
            FROM screenshot_tweet_matches m
            LEFT JOIN twitter_tweets t ON t.tweet_id = m.tweet_id
            LEFT JOIN twitter_likes l ON l.tweet_id = m.tweet_id
            WHERE m.screenshot_id = %(id)s
              AND COALESCE(t.full_text, l.full_text) IS NOT NULL
            ORDER BY m.score DESC
            """,
            {
                "id": id,
            },
        ).fetchall()

    lines = [f"Tweet ID {row['id']}"]
    if row["tweet_time"]:
        lines.append(f"Tweet time: {row['tweet_time'].isoformat()}")
    if row["mentioned_users"]:
        lines.append(f"Users: {', '.join('@' + u for u in row['mentioned_users'])}")
    lines.append("")
    lines.append(row["ocr_text_clean"] or "(no text)")

    description = row["image_description"]
    if description and not description.startswith("[error:"):
        lines.append("")
        lines.append(f"VLM transcription ({row['image_type'] or 'unclassified'}):")
        lines.append(description)

    shown = matches[:3]
    if shown:
        lines.append("")
        lines.append("Matched tweets from the Twitter export (exact text, no OCR errors):")
        for m in shown:
            label = "authored by" if m["authored"] else "liked via"
            header = f"[{label} @{m['account']}] tweet {m['tweet_id']}"
            if m["tweet_time"]:
                header += f" | {m['tweet_time'].isoformat()}"
            header += f" | match score {m['score']:.0f}"
            lines.append(header)
            lines.append(html.unescape(m["full_text"]))
        if len(matches) > len(shown):
            lines.append(f"(+{len(matches) - len(shown)} more matches)")

    return "\n".join(lines)


def _format_row(row) -> str:
    """Format a timeline row as plain text."""
    parts = [f"[ID {row['id']}]"]
    if row["tweet_time"]:
        parts.append(row["tweet_time"].isoformat())
    if row["mentioned_users"]:
        parts.append(", ".join("@" + u for u in dedup_handles(row["mentioned_users"])))
    snippet = (row["ocr_text_clean"] or "(no text)")[:SNIPPET_MAX_CHARS]
    parts.append(snippet)
    return " | ".join(parts)


@mcp.tool()
def nearby_screenshots(
    id: int,
    before: int = 5,
    after: int = 5,
) -> str:
    """Show screenshots captured around the same time as a given tweet. Not a
    search — just chronological neighbors. Requires a known ID from another tool.

    Args:
        id: Screenshot ID to center on.
        before: Number of earlier screenshots to show (default 5).
        after: Number of later screenshots to show (default 5).
    """
    with get_conn() as conn:
        before_rows, focal, after_rows = get_timeline_neighbors(conn, id, before=before, after=after)

    if focal is None:
        return f"No screenshot with id {id}"
    if focal["created_at"] is None:
        return _format_row(focal)

    lines = [_format_row(row) for row in before_rows]
    lines.append(f">>> {_format_row(focal)} <<<")
    lines.extend(_format_row(row) for row in after_rows)

    return "\n".join(lines)


@mcp.tool()
def find_related(id: int, limit: int = 10) -> str:
    """Find tweets with similar wording to a specific tweet. Use to find other
    parts of the same thread, conversation, or reply chain.

    Args:
        id: Screenshot ID to find related tweets for.
        limit: Maximum number of results (default 10).
    """
    matches = query_related(server._lsh, server._minhashes, id, top_n=limit)

    if not matches:
        return f"No related tweets found for ID {id}"

    match_ids = [mid for mid, _ in matches]
    sim_by_id = dict(matches)

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                id,
                ocr_text_clean,
                tweet_time,
                mentioned_users
            FROM screenshots
            WHERE id = ANY(%(ids)s)
            """,
            {
                "ids": match_ids,
            },
        ).fetchall()

    row_by_id = {row["id"]: row for row in rows}

    lines = []
    for mid in match_ids:
        row = row_by_id.get(mid)
        if not row:
            continue
        sim = sim_by_id[mid]
        parts = [f"[ID {mid}] sim={sim:.2f}"]
        if row["tweet_time"]:
            parts.append(row["tweet_time"].isoformat())
        if row["mentioned_users"]:
            parts.append(", ".join("@" + u for u in row["mentioned_users"]))
        snippet = (row["ocr_text_clean"] or "(no text)")[:SNIPPET_MAX_CHARS]
        parts.append(snippet)
        lines.append(" | ".join(parts))

    return "\n".join(lines)


@mcp.tool()
def search_by_user(
    handle: str,
    limit: int = 20,
    offset: int = 0,
    after: str | None = None,
    before: str | None = None,
    sort: str = "newest",
) -> str:
    """Find tweets mentioning a specific user. Use to follow what @someone was
    saying or being discussed.

    Args:
        handle: Twitter handle to search for (with or without @).
        limit: Max results to return (default 20).
        offset: Number of results to skip (default 0). Use to paginate.
        after: Only include tweets after this date (YYYY-MM-DD).
        before: Only include tweets before this date (YYYY-MM-DD).
        sort: "newest" (default) or "oldest".
    """
    handle = normalize_handle(handle)
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    conditions = ["%(handle)s = ANY(mentioned_users)"]
    params: dict = {
        "handle": handle,
        "limit": limit,
        "offset": offset,
    }
    add_time_filter(conditions, params, after, before)
    where = " AND ".join(conditions)

    if sort == "oldest":
        order_by = "COALESCE(tweet_time, created_at) ASC"
        sort_label = "oldest first"
    else:
        order_by = "COALESCE(tweet_time, created_at) DESC"
        sort_label = "newest first"

    with get_conn() as conn:
        stats = conn.execute(
            f"""
            SELECT
                count(*) AS tweet_count,
                min(COALESCE(tweet_time, created_at)) AS earliest,
                max(COALESCE(tweet_time, created_at)) AS latest
            FROM screenshots
            WHERE {where}
            """,
            params,
        ).fetchone()
        total = stats["tweet_count"]
        earliest = stats["earliest"]
        latest = stats["latest"]

        rows = conn.execute(
            f"""
            SELECT
                id,
                ocr_text_clean,
                tweet_time,
                mentioned_users
            FROM screenshots
            WHERE {where}
            ORDER BY {order_by}
            LIMIT %(limit)s
            OFFSET %(offset)s
            """,
            params,
        ).fetchall()

    if not rows:
        return f"No tweets found mentioning @{handle}"

    date_span = ""
    if earliest and latest:
        date_span = f"{earliest.strftime('%Y-%m-%d')} — {latest.strftime('%Y-%m-%d')}"

    start = offset + 1
    end = offset + len(rows)
    if len(rows) < total - offset:
        header = f"{total} tweets mentioning @{handle} ({date_span}, results {start}–{end} {sort_label})"
    else:
        header = f"{total} tweets mentioning @{handle} ({date_span}, {sort_label})"
    lines = [header + "\n"]
    for row in rows:
        lines.append(_format_row(row))

    return "\n".join(lines)


@mcp.tool()
def interactions(
    user1: str,
    user2: str,
    limit: int = 20,
    offset: int = 0,
    after: str | None = None,
    before: str | None = None,
) -> str:
    """Find tweets where two users appear together — conversations, quote
    tweets, and reply chains.

    Args:
        user1: First Twitter handle (with or without @).
        user2: Second Twitter handle (with or without @).
        limit: Max results to return (default 20).
        offset: Number of results to skip (default 0). Use to paginate.
        after: Only include tweets after this date (YYYY-MM-DD).
        before: Only include tweets before this date (YYYY-MM-DD).
    """
    user1 = normalize_handle(user1)
    user2 = normalize_handle(user2)
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    conditions = ["mentioned_users @> ARRAY[%(u1)s, %(u2)s]"]
    params: dict = {
        "u1": user1,
        "u2": user2,
        "limit": limit,
        "offset": offset,
    }
    add_time_filter(conditions, params, after, before)
    where = " AND ".join(conditions)

    with get_conn() as conn:
        total = conn.execute(
            f"SELECT count(*) FROM screenshots WHERE {where}",
            params,
        ).fetchone()["count"]

        rows = conn.execute(
            f"""
            SELECT
                id,
                ocr_text_clean,
                tweet_time,
                mentioned_users
            FROM screenshots
            WHERE {where}
            ORDER BY COALESCE(tweet_time, created_at) DESC
            LIMIT %(limit)s
            OFFSET %(offset)s
            """,
            params,
        ).fetchall()

    if not rows:
        return f"No tweets found with both @{user1} and @{user2}"

    start = offset + 1
    end = offset + len(rows)
    if len(rows) < total - offset:
        header = f"Results {start}–{end} of {total} tweets with @{user1} + @{user2} (newest first)"
    else:
        header = f"Found {total} tweets with @{user1} + @{user2} (newest first)"
    lines = [header + "\n"]
    for row in rows:
        lines.append(_format_row(row))

    return "\n".join(lines)
