"""Import Twitter data export (likes + tweets) into PostgreSQL."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import orjson

from .db import check_db, get_conn


def _parse_export_js(path):
    """Strip the window.YTD.*.part0 = prefix and parse JSON with orjson."""
    raw = path.read_bytes()
    # Find the first '=' and skip past it
    idx = raw.index(b"=")
    return orjson.loads(raw[idx + 1 :])


def _read_account(export_dir):
    """Return the Twitter username from account.js."""
    data = _parse_export_js(export_dir / "data" / "account.js")
    return data[0]["account"]["username"]


def _import_likes(conn, export_dir, account):
    """Parse like.js and upsert into twitter_likes."""
    data = _parse_export_js(export_dir / "data" / "like.js")
    count = 0
    for item in data:
        like = item["like"]
        conn.execute(
            """
            INSERT INTO twitter_likes (
                tweet_id,
                full_text,
                expanded_url,
                account
            ) VALUES (
                %(tweet_id)s,
                %(full_text)s,
                %(expanded_url)s,
                %(account)s
            )
            ON CONFLICT (tweet_id) DO UPDATE SET
                full_text = EXCLUDED.full_text,
                expanded_url = EXCLUDED.expanded_url,
                account = EXCLUDED.account
            """,
            {
                "tweet_id": int(like["tweetId"]),
                "full_text": like.get("fullText"),
                "expanded_url": like.get("expandedUrl"),
                "account": account,
            },
        )
        count += 1
    conn.commit()
    return count


def _import_tweets(conn, export_dir, account):
    """Parse tweets.js and upsert into twitter_tweets."""
    data = _parse_export_js(export_dir / "data" / "tweets.js")
    count = 0
    for item in data:
        tw = item["tweet"]
        created_at = datetime.strptime(tw["created_at"], "%a %b %d %H:%M:%S %z %Y")
        conn.execute(
            """
            INSERT INTO twitter_tweets (
                tweet_id,
                full_text,
                created_at,
                in_reply_to_tweet_id,
                in_reply_to_user_id,
                in_reply_to_screen_name,
                retweet_count,
                favorite_count,
                lang,
                account
            ) VALUES (
                %(tweet_id)s,
                %(full_text)s,
                %(created_at)s,
                %(in_reply_to_tweet_id)s,
                %(in_reply_to_user_id)s,
                %(in_reply_to_screen_name)s,
                %(retweet_count)s,
                %(favorite_count)s,
                %(lang)s,
                %(account)s
            )
            ON CONFLICT (tweet_id) DO UPDATE SET
                full_text = EXCLUDED.full_text,
                created_at = EXCLUDED.created_at,
                in_reply_to_tweet_id = EXCLUDED.in_reply_to_tweet_id,
                in_reply_to_user_id = EXCLUDED.in_reply_to_user_id,
                in_reply_to_screen_name = EXCLUDED.in_reply_to_screen_name,
                retweet_count = EXCLUDED.retweet_count,
                favorite_count = EXCLUDED.favorite_count,
                lang = EXCLUDED.lang,
                account = EXCLUDED.account
            """,
            {
                "tweet_id": int(tw["id_str"]),
                "full_text": tw.get("full_text"),
                "created_at": created_at,
                "in_reply_to_tweet_id": int(tw["in_reply_to_status_id"]) if "in_reply_to_status_id" in tw else None,
                "in_reply_to_user_id": int(tw["in_reply_to_user_id"]) if "in_reply_to_user_id" in tw else None,
                "in_reply_to_screen_name": tw.get("in_reply_to_screen_name"),
                "retweet_count": int(tw.get("retweet_count", 0)),
                "favorite_count": int(tw.get("favorite_count", 0)),
                "lang": tw.get("lang"),
                "account": account,
            },
        )
        count += 1
    conn.commit()
    return count


def main():
    parser = argparse.ArgumentParser(description="Import a Twitter data export")
    parser.add_argument("export_dir", type=Path, help="Path to the Twitter export directory")
    args = parser.parse_args()

    export_dir = args.export_dir.expanduser().resolve()
    if not (export_dir / "data" / "account.js").exists():
        print(f"Error: {export_dir} does not look like a Twitter export (no data/account.js)", file=sys.stderr)
        sys.exit(1)

    check_db()
    account = _read_account(export_dir)
    print(f"Importing export for @{account}")

    with get_conn() as conn:
        likes = _import_likes(conn, export_dir, account)
        print(f"  Likes: {likes}")

        tweets = _import_tweets(conn, export_dir, account)
        print(f"  Tweets: {tweets}")

    print("Done.")
