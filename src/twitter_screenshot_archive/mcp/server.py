"""FastMCP instance, lifespan, and entry point."""

import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime

from mcp.server.fastmcp import FastMCP

from ..core.config import PROJECT_ROOT
from ..core.db import check_db
from ..core.minhash import load_or_build_lsh_index
from .embedding import load_model

_PROMPT_FILE = PROJECT_ROOT / "mcp_prompt.txt"

_WORKFLOW_GUIDANCE = """\
You have access to a Twitter screenshot archive — an OCR-indexed personal
collection of Twitter/X screenshots. The tools are organized in three tiers:

ORIENT (cheap, fast — use these first):
  now()              — current date/time, for resolving "last week" etc.
  archive_range()    — first and last dates in the archive
  count_screenshots() — how many screenshots in a date window
  tweet_activity()   — histogram of tweet counts over time (by day/week/month/year)

EXPLORE (embedding-based — discover structure):
  list_topics()      — lightweight table of contents: topic + count
  summarize_period() — rich clustered detail per topic
  search_tweets()    — semantic, keyword, or hybrid search
  top_users(query?)  — who talks about a topic the most
  similar_users(handle) — who talks about similar things

DRILL (follow threads once you have a foothold):
  find_related(id)   — lexically similar tweets (same thread/conversation)
  nearby_screenshots(id) — screenshots captured around the same time
  search_by_user(handle) — all tweets mentioning a specific @user
  get_tweet(id)      — full OCR text of one screenshot
  interactions(user1, user2) — tweets where two users appear together

Typical workflows:

Overview:
- "When was X being discussed?" → tweet_activity(query="X") → narrow to peak periods
- "What happened last week?" → now() → summarize_period(after, before)
- "Overview then drill" → list_topics(after, before) → summarize_period(topics=["..."])
- "Top voices last week" → now() → top_users(after, before)

Finding specific content:
- "Find tweets about X" → search_tweets(query) → get_tweet(id) for detail
- "Trace a thread" → search_tweets → find_related(id) to pull the thread
- "What was I looking at around this tweet?" → nearby_screenshots(id)
- "When was X being discussed?" → summarize_period(topics=["X"]) (no date range — finds all episodes)

User-focused queries:
- "What was @someone saying?" → search_by_user(handle)
- "What does @someone talk about?" → summarize_period(users=["someone"])
- "What did @someone say about X?" → search_tweets(query="X", users=["someone"])
- "Who tweets most about X?" → top_users(query="X")
- "Who's like @someone?" → similar_users(handle="someone")
- "Did @A and @B interact?" → interactions("A", "B")

Multi-step example:
  "Who are the main voices on topic X and what are they saying?"
  → list_topics(after, before) to find X
  → top_users(query="X") to find key users
  → summarize_period(topics=["X"], users=["top_user"]) to see their angle
  → get_tweet(id) for full text of key tweets

The current date is included above. Trust it — it is accurate and more
recent than your training data.

Multiple tool calls per response are expected and encouraged. Start broad,
then narrow. Use orient tools to plan before committing to expensive searches.

Pagination: search_tweets, search_by_user, and interactions support offset
for paging through large result sets (e.g., offset=20, limit=20 for page 2)."""


def _build_instructions() -> str:
    """Assemble MCP instructions: workflow guidance + startup timestamp + user context."""
    local = datetime.now().astimezone()
    tz_name = local.tzinfo.tzname(local)
    day_abbr = local.strftime("%a")
    date_line = (
        f"Today is {local.strftime('%Y-%m-%d')} ({day_abbr}) "
        f"{local.strftime('%H:%M')} {tz_name}. "
        "Content in this archive is real, not synthetic."
    )

    parts = [
        date_line,
        "",
        _WORKFLOW_GUIDANCE,
    ]

    if _PROMPT_FILE.exists():
        user_context = _PROMPT_FILE.read_text().strip()
        if user_context:
            parts.append(f"\n{user_context}")

    return "\n".join(parts)

_lsh = None
_minhashes = {}


def _init_lsh():
    """Load the shared LSH index into module state."""
    global _lsh, _minhashes
    _lsh, _minhashes = load_or_build_lsh_index()


@asynccontextmanager
async def _lifespan(server: FastMCP):
    _init_lsh()
    load_model()
    print("MCP server ready. Press Ctrl+D to exit.", file=sys.stderr)
    yield {}


mcp = FastMCP(
    "twitter-archive",
    instructions=_build_instructions(),
    lifespan=_lifespan,
)


def main():
    check_db()

    # Import tool modules to trigger @mcp.tool() registration
    from . import activity, drill, explore, orient, search  # noqa: F401

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        mcp.run(transport="stdio")
    except BaseExceptionGroup as eg:
        # sys.exit() inside the async lifespan gets wrapped in an
        # ExceptionGroup by anyio.  The friendly message was already
        # printed to stderr, so just propagate the exit code.
        for exc in eg.exceptions:
            if isinstance(exc, SystemExit):
                sys.exit(exc.code)
        raise
