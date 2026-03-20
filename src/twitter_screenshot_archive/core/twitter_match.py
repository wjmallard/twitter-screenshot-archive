"""Match liked/authored tweets to screenshots via word n-gram seeding + rapidfuzz."""

import html
import re
import sys
from collections import defaultdict

from rapidfuzz import fuzz
from tqdm import tqdm

from .db import check_db, get_conn

SCORE_THRESHOLD = 80
TOP_CANDIDATES = 20

_TCO_RE = re.compile(r"https?://t\.co/\S+")


def _norm(s):
    """Whitespace-normalize and lowercase."""
    return " ".join(s.lower().split()) if s else ""


def _clean_tweet_text(text):
    """Decode HTML entities and strip t.co URLs."""
    text = html.unescape(text)
    text = _TCO_RE.sub("", text)
    return text


def _word_ngrams(text, n=3):
    """Word-level n-grams from normalized text."""
    words = text.split()
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def _build_index(conn):
    """Build in-memory word 3-gram inverted index from screenshots.

    Returns (index, ocr_map) where:
      index:   dict[str, set[int]]  — ngram → screenshot IDs
      ocr_map: dict[int, str]       — screenshot ID → normalized OCR text
    """
    rows = conn.execute(
        "SELECT id, ocr_text_clean FROM screenshots "
        "WHERE ocr_text_clean IS NOT NULL AND ocr_text_clean != ''"
    ).fetchall()
    print(f"Loaded {len(rows)} screenshots", file=sys.stderr)

    index = defaultdict(set)
    ocr_map = {}
    for sid, ocr in rows:
        nocr = _norm(ocr)
        ocr_map[sid] = nocr
        for ng in _word_ngrams(nocr):
            index[ng].add(sid)

    print(f"Built index: {len(index):,} unique 3-grams", file=sys.stderr)
    return index, ocr_map


def _match_source(conn, index, ocr_map, source_table, source_label):
    """Match tweets from a source table against the screenshot index.

    Inserts all matches above SCORE_THRESHOLD into screenshot_tweet_matches.
    Returns (matched_tweets, total_tweets, total_links) counts.
    """
    if source_table == "twitter_likes":
        rows = conn.execute(
            "SELECT tweet_id, full_text FROM twitter_likes "
            "WHERE full_text NOT LIKE '%%{learnmore}%%' "
            "  AND full_text NOT LIKE 'This Post is from%%' "
            "  AND length(full_text) > 30"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT tweet_id, full_text FROM twitter_tweets "
            "WHERE full_text IS NOT NULL AND length(full_text) > 30"
        ).fetchall()

    matched_tweets = 0
    total_links = 0
    progress = tqdm(rows, desc=source_label, file=sys.stderr)
    for tweet_id, full_text in progress:
        needle = _norm(_clean_tweet_text(full_text))
        ngrams = _word_ngrams(needle)
        if not ngrams:
            continue

        # Seed: count shared n-grams per screenshot
        hits = defaultdict(int)
        for ng in ngrams:
            for sid in index.get(ng, ()):
                hits[sid] += 1

        if not hits:
            continue

        # Top candidates by shared n-gram count
        top = sorted(hits.items(), key=lambda x: x[1], reverse=True)[:TOP_CANDIDATES]

        # Score each candidate with rapidfuzz partial_ratio
        tweet_matched = False
        for sid, _count in top:
            score = fuzz.partial_ratio(needle, ocr_map[sid])
            if score >= SCORE_THRESHOLD:
                conn.execute(
                    """
                    INSERT INTO screenshot_tweet_matches (
                        screenshot_id,
                        tweet_id,
                        score
                    ) VALUES (
                        %(screenshot_id)s,
                        %(tweet_id)s,
                        %(score)s
                    )
                    ON CONFLICT (screenshot_id, tweet_id) DO UPDATE SET
                        score = EXCLUDED.score
                    """,
                    {
                        "screenshot_id": sid,
                        "tweet_id": tweet_id,
                        "score": score,
                    },
                )
                total_links += 1
                tweet_matched = True

        if tweet_matched:
            matched_tweets += 1

    conn.commit()
    progress.close()
    return matched_tweets, len(rows), total_links


def main():
    check_db()

    with get_conn() as conn:
        print("Building n-gram index...", file=sys.stderr)
        index, ocr_map = _build_index(conn)

        likes_matched, likes_total, likes_links = _match_source(
            conn, index, ocr_map, "twitter_likes", "Likes"
        )
        tweets_matched, tweets_total, tweets_links = _match_source(
            conn, index, ocr_map, "twitter_tweets", "Tweets"
        )

    print(f"\nLikes:  {likes_matched}/{likes_total} matched ({likes_links} links)", file=sys.stderr)
    print(f"Tweets: {tweets_matched}/{tweets_total} matched ({tweets_links} links)", file=sys.stderr)
