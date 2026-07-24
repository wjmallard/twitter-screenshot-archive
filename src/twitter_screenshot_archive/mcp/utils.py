"""Shared MCP utilities."""

from ..core.db import get_conn


def add_time_filter(
    conditions: list[str],
    params: dict,
    after: str | None,
    before: str | None,
    column: str = "COALESCE(tweet_time, created_at)",
) -> None:
    """Append after/before date conditions and params in place."""
    if after:
        conditions.append(f"{column} >= %(after)s::date")
        params["after"] = after
    if before:
        conditions.append(f"{column} < %(before)s::date")
        params["before"] = before


def add_users_filter(
    conditions: list[str],
    params: dict,
    users: list[str] | None,
) -> None:
    """Append a mentioned_users overlap condition and param in place."""
    if users:
        conditions.append("mentioned_users && %(users)s::text[]")
        params["users"] = normalize_handles(users)


def normalize_handle(handle: str) -> str:
    """Strip a leading @ and lowercase."""
    return handle.lstrip("@").lower()


def normalize_handles(handles: list[str]) -> list[str]:
    return [normalize_handle(h) for h in handles]


def validate_keywords(keywords: str) -> bool:
    """Check tsquery syntax by parsing it server-side."""
    try:
        with get_conn() as conn:
            conn.execute(
                "SELECT to_tsquery('english', %(kw)s)",
                {"kw": keywords},
            )
        return True
    except Exception:
        return False


def merge_similar_handles(
    user_counts: dict[str, int],
    primary: str | None = None,
) -> dict[str, int]:
    """Group handles by prefix overlap, keeping the longest variant.

    OCR frequently truncates handles — @bonzerba, @bon, @bonzerb are all
    fragments of @bonzerbarry.  This merges their counts into the longest
    matching handle.

    When *primary* is provided, also excludes any handle that is a prefix of
    or prefixed by the primary handle (catches both the full handle and its
    OCR fragments).
    """
    # Process longest first — the longest variant becomes canonical
    sorted_handles = sorted(user_counts, key=len, reverse=True)

    merged: dict[str, int] = {}

    for h in sorted_handles:
        # Skip fragments of the primary handle
        if primary and (h.startswith(primary) or primary.startswith(h)):
            continue

        # Check if h is a prefix of any existing canonical handle
        matched = False
        for canonical in merged:
            if canonical.startswith(h):
                merged[canonical] += user_counts[h]
                matched = True
                break

        if not matched:
            merged[h] = user_counts[h]

    return merged


def dedup_handles(handles: list[str] | None) -> list[str]:
    """Deduplicate a handle list by prefix overlap, keeping the longest."""
    if not handles:
        return []

    result: list[str] = []
    for h in sorted(set(handles), key=len, reverse=True):
        if not any(existing.startswith(h) for existing in result):
            result.append(h)

    return result
