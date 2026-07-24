import sys
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

from . import config

DB_NAME = "twitter_screenshot_archive"


def check_db():
    """Verify PostgreSQL is reachable. Call once at startup."""
    try:
        psycopg.connect(dbname=DB_NAME).close()
    except psycopg.OperationalError as exc:
        print(f"Error: Could not connect to PostgreSQL: {exc}", file=sys.stderr)
        print("Is the server running? Try: brew services start postgresql@17", file=sys.stderr)
        sys.exit(1)


@contextmanager
def get_conn():
    conn = psycopg.connect(dbname=DB_NAME, row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_screenshot(conn, row):
    """Insert or update one screenshot. Expects the dict built by process_image()."""
    conn.execute(
        """
        INSERT INTO screenshots (
            file_path,
            ocr_text,
            ocr_text_clean,
            created_at,
            created_at_local,
            timezone,
            width,
            height,
            file_size,
            minhash_signature,
            mentioned_users,
            tweet_time,
            tweet_time_source
        ) VALUES (
            %(file_path)s,
            %(ocr_text)s,
            %(ocr_text_clean)s,
            %(created_at)s,
            %(created_at_local)s,
            %(timezone)s,
            %(width)s,
            %(height)s,
            %(file_size)s,
            %(minhash_signature)s,
            %(mentioned_users)s,
            %(tweet_time)s,
            %(tweet_time_source)s
        )
        ON CONFLICT (file_path) DO UPDATE SET
            ocr_text = EXCLUDED.ocr_text,
            ocr_text_clean = EXCLUDED.ocr_text_clean,
            created_at = EXCLUDED.created_at,
            created_at_local = EXCLUDED.created_at_local,
            timezone = EXCLUDED.timezone,
            width = EXCLUDED.width,
            height = EXCLUDED.height,
            file_size = EXCLUDED.file_size,
            minhash_signature = EXCLUDED.minhash_signature,
            mentioned_users = EXCLUDED.mentioned_users,
            tweet_time = EXCLUDED.tweet_time,
            tweet_time_source = EXCLUDED.tweet_time_source
        """,
        row,
    )


def images_in_db(conn) -> set[str]:
    rows = conn.execute("SELECT file_path FROM screenshots").fetchall()
    return {row["file_path"] for row in rows}


_HALF_LIFE_SECS = config.DECAY_HALF_LIFE_DAYS * 86400
_DECAY = f"(1.0 / (EXTRACT(EPOCH FROM now() - COALESCE(created_at, now())) / {_HALF_LIFE_SECS} + 1))"

_FT_SCORE = "ts_rank(ocr_text_tsv, websearch_to_tsquery('english', %(query)s))"
_TG_SCORE = "word_similarity(%(query)s, ocr_text)"

SORT_OPTIONS = {
    "best": {"word": f"{_FT_SCORE} * {_DECAY} DESC", "char": f"{_TG_SCORE} * {_DECAY} DESC", "none": f"{_DECAY} DESC"},
    "strongest": {"word": f"{_FT_SCORE} DESC", "char": f"{_TG_SCORE} DESC", "none": "created_at DESC NULLS LAST"},
    "newest": "created_at DESC NULLS LAST",
    "oldest": "created_at ASC NULLS LAST",
}


def _resolve_sort(sort, fuzzy):
    if sort not in SORT_OPTIONS:
        sort = "best"
    opt = SORT_OPTIONS[sort]
    return opt[fuzzy] if isinstance(opt, dict) else opt


def search_fulltext(conn, query, limit=50, offset=0, sort="best"):
    order = _resolve_sort(sort, "word")
    return conn.execute(
        f"""
        SELECT
            id,
            file_path,
            ocr_text,
            created_at_local,
            timezone,
            width,
            height,
            file_size,
            ts_rank(ocr_text_tsv, websearch_to_tsquery('english', %(query)s)) AS score
        FROM screenshots
        WHERE ocr_text_tsv @@ websearch_to_tsquery('english', %(query)s)
        ORDER BY {order}
        LIMIT %(limit)s
        OFFSET %(offset)s
        """,
        {
            "query": query,
            "limit": limit,
            "offset": offset,
        },
    ).fetchall()


def search_trigram(conn, query, limit=50, offset=0, sort="best"):
    order = _resolve_sort(sort, "char")
    return conn.execute(
        f"""
        SELECT
            id,
            file_path,
            ocr_text,
            created_at_local,
            timezone,
            width,
            height,
            file_size,
            word_similarity(%(query)s, ocr_text) AS score
        FROM screenshots
        WHERE %(query)s <<%% ocr_text
        ORDER BY {order}
        LIMIT %(limit)s
        OFFSET %(offset)s
        """,
        {
            "query": query,
            "limit": limit,
            "offset": offset,
        },
    ).fetchall()


def _like_pattern(query):
    """Escape LIKE wildcards so user input matches literally."""
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def search_exact(conn, query, limit=50, offset=0, sort="best"):
    order = _resolve_sort(sort, "none")
    return conn.execute(
        f"""
        SELECT
            id,
            file_path,
            ocr_text,
            created_at_local,
            timezone,
            width,
            height,
            file_size,
            1.0 AS score
        FROM screenshots
        WHERE ocr_text ILIKE %(pattern)s
        ORDER BY {order}
        LIMIT %(limit)s
        OFFSET %(offset)s
        """,
        {
            "pattern": _like_pattern(query),
            "limit": limit,
            "offset": offset,
        },
    ).fetchall()


def count_fulltext(conn, query):
    row = conn.execute(
        """
        SELECT count(*)
        FROM screenshots
        WHERE ocr_text_tsv @@ websearch_to_tsquery('english', %(query)s)
        """,
        {
            "query": query,
        },
    ).fetchone()
    return row["count"]


def count_trigram(conn, query):
    row = conn.execute(
        """
        SELECT count(*)
        FROM screenshots
        WHERE %(query)s <<%% ocr_text
        """,
        {
            "query": query,
        },
    ).fetchone()
    return row["count"]


def count_exact(conn, query):
    row = conn.execute(
        """
        SELECT count(*)
        FROM screenshots
        WHERE ocr_text ILIKE %(pattern)s
        """,
        {
            "pattern": _like_pattern(query),
        },
    ).fetchone()
    return row["count"]


def count_screenshots(conn):
    row = conn.execute("SELECT count(*) FROM screenshots").fetchone()
    return row["count"]


def signature_fingerprint(conn):
    """Return (count, max_id) for cache invalidation of the LSH index."""
    row = conn.execute(
        """
        SELECT
            count(*) AS sig_count,
            coalesce(max(id), 0) AS max_id
        FROM screenshots
        WHERE minhash_signature IS NOT NULL
        """
    ).fetchone()
    return (row["sig_count"], row["max_id"])


def load_all_signatures(conn):
    """Load all (id, minhash_signature) pairs for LSH index building."""
    return conn.execute(
        "SELECT id, minhash_signature FROM screenshots WHERE minhash_signature IS NOT NULL"
    ).fetchall()


# Columns needed by both consumers of get_timeline_neighbors: the web
# timeline (file metadata) and the MCP nearby_screenshots tool (tweet text).
_NEIGHBOR_COLUMNS = """
            id,
            file_path,
            ocr_text,
            ocr_text_clean,
            created_at,
            created_at_local,
            timezone,
            width,
            height,
            file_size,
            tweet_time,
            mentioned_users
""".strip()


def get_timeline_neighbors(conn, screenshot_id, before=1, after=1):
    """Get screenshots around a given screenshot in capture-time order.

    Returns (before_rows, focal_row, after_rows) of screenshot row dicts.
    """
    focal = conn.execute(
        f"""
        SELECT
            {_NEIGHBOR_COLUMNS}
        FROM screenshots
        WHERE id = %(id)s
        """,
        {
            "id": screenshot_id,
        },
    ).fetchone()
    if not focal:
        return [], None, []

    if focal["created_at"] is None:
        return [], focal, []

    before_rows = conn.execute(
        f"""
        SELECT
            {_NEIGHBOR_COLUMNS}
        FROM screenshots
        WHERE (created_at, id) < (%(ts)s, %(id)s)
          AND created_at IS NOT NULL
        ORDER BY created_at DESC, id DESC
        LIMIT %(limit)s
        """,
        {
            "ts": focal["created_at"],
            "id": screenshot_id,
            "limit": before,
        },
    ).fetchall()

    after_rows = conn.execute(
        f"""
        SELECT
            {_NEIGHBOR_COLUMNS}
        FROM screenshots
        WHERE (created_at, id) > (%(ts)s, %(id)s)
          AND created_at IS NOT NULL
        ORDER BY created_at ASC, id ASC
        LIMIT %(limit)s
        """,
        {
            "ts": focal["created_at"],
            "id": screenshot_id,
            "limit": after,
        },
    ).fetchall()

    return list(reversed(before_rows)), focal, list(after_rows)


def get_screenshots_by_ids(conn, ids):
    """Fetch screenshot details for a list of IDs. Returns dict of {id: row_dict}."""
    if not ids:
        return {}
    rows = conn.execute(
        """
        SELECT
            id,
            file_path,
            ocr_text,
            created_at_local,
            timezone,
            width,
            height,
            file_size
        FROM screenshots
        WHERE id = ANY(%(ids)s)
        """,
        {
            "ids": list(ids),
        },
    ).fetchall()
    return {row["id"]: row for row in rows}
