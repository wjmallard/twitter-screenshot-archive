"""Orientation tools — cheap, fast, no embeddings."""

from datetime import datetime, timezone

from ..core.db import get_conn
from .server import mcp
from .utils import add_time_filter


@mcp.tool()
async def now() -> str:
    """Get the current date and time. Use this to resolve relative
    references like 'last week' or 'yesterday'."""
    utc = datetime.now(timezone.utc)
    local = datetime.now().astimezone()
    tz_name = local.tzinfo.tzname(local)
    return (
        f"UTC: {utc.isoformat()}\n"
        f"Local: {local.isoformat()} ({tz_name})"
    )


@mcp.tool()
async def archive_range() -> str:
    """Get the first and last dates in the archive."""
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT
                min(created_at) AS first,
                max(created_at) AS last
            FROM screenshots
            """
        ).fetchone()

    if not row or not row["first"]:
        return "Archive is empty."

    return (
        f"First: {row['first'].isoformat()}\n"
        f"Last: {row['last'].isoformat()}"
    )


@mcp.tool()
async def count_screenshots(
    after: str | None = None,
    before: str | None = None,
) -> str:
    """Count screenshots in a time window. Use to gauge density before
    searching.

    Args:
        after: Only count screenshots after this date (YYYY-MM-DD).
        before: Only count screenshots before this date (YYYY-MM-DD).
    """
    conditions = []
    params: dict = {}
    add_time_filter(conditions, params, after, before, column="created_at")

    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    with get_conn() as conn:
        row = conn.execute(
            f"SELECT count(*) FROM screenshots {where}",
            params,
        ).fetchone()

    return str(row["count"])
