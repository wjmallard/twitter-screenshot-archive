import sys

import psycopg
from contextlib import contextmanager

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
    conn = psycopg.connect(dbname=DB_NAME)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_screenshot(conn, file_path, ocr_text, created_at, created_at_local, timezone, width, height,
                      file_size=None, minhash_signature=None, mentioned_users=None,
                      tweet_time=None, tweet_time_source=None, ocr_text_clean=None):
    conn.execute(
        """
        INSERT INTO screenshots (file_path, ocr_text, ocr_text_clean, created_at, created_at_local, timezone,
                                 width, height, file_size, minhash_signature, mentioned_users,
                                 tweet_time, tweet_time_source)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
        (file_path, ocr_text, ocr_text_clean, created_at, created_at_local, timezone, width, height,
         file_size, minhash_signature, mentioned_users, tweet_time, tweet_time_source),
    )


def images_in_db(conn) -> set[str]:
    rows = conn.execute("SELECT file_path FROM screenshots").fetchall()
    return {row[0] for row in rows}


_HALF_LIFE_SECS = config.DECAY_HALF_LIFE_DAYS * 86400
_DECAY = f"(1.0 / (EXTRACT(EPOCH FROM now() - COALESCE(created_at, now())) / {_HALF_LIFE_SECS} + 1))"

_FT_SCORE = "ts_rank(ocr_text_tsv, websearch_to_tsquery('english', %s))"
_TG_SCORE = "word_similarity(%s, ocr_text)"

SORT_OPTIONS = {
    "best": {"word": f"{_FT_SCORE} * {_DECAY} DESC", "char": f"{_TG_SCORE} * {_DECAY} DESC", "none": f"{_DECAY} DESC"},
    "strongest": {"word": f"{_FT_SCORE} DESC", "char": f"{_TG_SCORE} DESC", "none": "created_at DESC NULLS LAST"},
    "newest": "created_at DESC NULLS LAST",
    "oldest": "created_at ASC NULLS LAST",
}

# Number of extra %s params the ORDER BY clause needs for the query string
_SORT_EXTRA_PARAMS = {
    "best": {"word": 1, "char": 1, "none": 0},
    "strongest": {"word": 1, "char": 1, "none": 0},
    "newest": 0,
    "oldest": 0,
}


def _resolve_sort(sort, fuzzy):
    if sort not in SORT_OPTIONS:
        sort = "best"
    opt = SORT_OPTIONS[sort]
    order = opt[fuzzy] if isinstance(opt, dict) else opt
    extra_opt = _SORT_EXTRA_PARAMS[sort]
    extra = extra_opt[fuzzy] if isinstance(extra_opt, dict) else extra_opt
    return order, extra


def search_fulltext(conn, query, limit=50, offset=0, sort="best"):
    order, extra = _resolve_sort(sort, "word")
    params = (query, query) + (query,) * extra + (limit, offset)
    return conn.execute(
        f"""
        SELECT id, file_path, ocr_text, created_at_local, timezone, width, height, file_size,
               ts_rank(ocr_text_tsv, websearch_to_tsquery('english', %s)) AS score
        FROM screenshots
        WHERE ocr_text_tsv @@ websearch_to_tsquery('english', %s)
        ORDER BY {order}
        LIMIT %s OFFSET %s
        """,
        params,
    ).fetchall()


def search_trigram(conn, query, limit=50, offset=0, sort="best"):
    order, extra = _resolve_sort(sort, "char")
    params = (query, query) + (query,) * extra + (limit, offset)
    return conn.execute(
        f"""
        SELECT id, file_path, ocr_text, created_at_local, timezone, width, height, file_size,
               word_similarity(%s, ocr_text) AS score
        FROM screenshots
        WHERE %s <<%% ocr_text
        ORDER BY {order}
        LIMIT %s OFFSET %s
        """,
        params,
    ).fetchall()


def search_exact(conn, query, limit=50, offset=0, sort="best"):
    order, extra = _resolve_sort(sort, "none")
    like_param = f"%{query}%"
    params = (like_param,) + (query,) * extra + (limit, offset)
    return conn.execute(
        f"""
        SELECT id, file_path, ocr_text, created_at_local, timezone, width, height, file_size,
               1.0 AS score
        FROM screenshots
        WHERE ocr_text ILIKE %s
        ORDER BY {order}
        LIMIT %s OFFSET %s
        """,
        params,
    ).fetchall()


def count_fulltext(conn, query):
    row = conn.execute(
        "SELECT count(*) FROM screenshots WHERE ocr_text_tsv @@ websearch_to_tsquery('english', %s)",
        (query,),
    ).fetchone()
    return row[0]


def count_trigram(conn, query):
    row = conn.execute(
        "SELECT count(*) FROM screenshots WHERE %s <<% ocr_text",
        (query,),
    ).fetchone()
    return row[0]


def count_exact(conn, query):
    row = conn.execute(
        "SELECT count(*) FROM screenshots WHERE ocr_text ILIKE %s",
        (f"%{query}%",),
    ).fetchone()
    return row[0]


def count_screenshots(conn):
    row = conn.execute("SELECT count(*) FROM screenshots").fetchone()
    return row[0]


def signature_fingerprint(conn):
    """Return (count, max_id) for cache invalidation of the LSH index."""
    row = conn.execute(
        "SELECT count(*), coalesce(max(id), 0) FROM screenshots WHERE minhash_signature IS NOT NULL"
    ).fetchone()
    return (row[0], row[1])


def load_all_signatures(conn):
    """Load all (id, minhash_signature) pairs for LSH index building."""
    return conn.execute(
        "SELECT id, minhash_signature FROM screenshots WHERE minhash_signature IS NOT NULL"
    ).fetchall()


def get_timeline_neighbors(conn, screenshot_id, before=1, after=1):
    """Get screenshots around a given screenshot in capture-time order.

    Returns (before_rows, focal_row, after_rows) where each row is
    (id, file_path, ocr_text, created_at_local, timezone, width, height, file_size).
    """
    focal = conn.execute(
        """
        SELECT id, file_path, ocr_text, created_at_local, timezone, width, height, file_size, created_at
        FROM screenshots WHERE id = %s
        """,
        (screenshot_id,),
    ).fetchone()
    if not focal:
        return [], None, []

    focal_time = focal[8]  # created_at
    if focal_time is None:
        return [], focal[:8], []

    before_rows = conn.execute(
        """
        SELECT id, file_path, ocr_text, created_at_local, timezone, width, height, file_size
        FROM screenshots
        WHERE (created_at, id) < (%s, %s) AND created_at IS NOT NULL
        ORDER BY created_at DESC, id DESC
        LIMIT %s
        """,
        (focal_time, screenshot_id, before),
    ).fetchall()

    after_rows = conn.execute(
        """
        SELECT id, file_path, ocr_text, created_at_local, timezone, width, height, file_size
        FROM screenshots
        WHERE (created_at, id) > (%s, %s) AND created_at IS NOT NULL
        ORDER BY created_at ASC, id ASC
        LIMIT %s
        """,
        (focal_time, screenshot_id, after),
    ).fetchall()

    return list(reversed(before_rows)), focal[:8], list(after_rows)


def get_screenshots_by_ids(conn, ids):
    """Fetch screenshot details for a list of IDs. Returns dict of {id: row_dict}."""
    if not ids:
        return {}
    rows = conn.execute(
        """
        SELECT id, file_path, ocr_text, created_at_local, timezone, width, height, file_size
        FROM screenshots
        WHERE id = ANY(%s)
        """,
        (list(ids),),
    ).fetchall()
    return {
        row[0]: {
            "file_path": row[1],
            "ocr_text": row[2],
            "created_at_local": row[3],
            "timezone": row[4],
            "width": row[5],
            "height": row[6],
            "file_size": row[7],
        }
        for row in rows
    }
