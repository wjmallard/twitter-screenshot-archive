# Twitter Screenshot Archive

Full-text search and topic discovery over a personal archive of Twitter screenshots. OCR-indexes images into Postgres with two optional interfaces: a Flask web UI for fuzzy search and similarity browsing, and an MCP server for LLM-driven semantic search, topic clustering, and discourse tracing.

## Overview

1. **Ingest** — OCR screenshots with Tesseract, compute MinHash signatures, insert into Postgres
2. **Search** (optional) — Flask web UI with fuzzy search, related-tweet discovery, and image lightbox
3. **MCP Server** (optional) — Semantic search, topic clustering, and discourse tracing via in-process MLX embeddings, exposed as MCP tools for LLM chat models (e.g. Claude, Qwen)
4. **Embed** (optional) — Standalone embedding backfill using Qwen3-Embedding on Apple Silicon via MLX
5. **Describe** (optional) — VLM image descriptions using Qwen2.5-VL-7B, classifying and transcribing tweet screenshots
6. **Import & Match** (optional) — Load a Twitter data export and link screenshots to their exact tweets

Dates are parsed from JSON sidecars with EXIF fallback.

## Stack

- Python 3.12
- PostgreSQL (pg_trgm, pgvector)
- Tesseract
- OpenCV
- datasketch
- Flask (optional — web UI)
- MLX, mlx-lm, mlx-vlm (optional — Apple Silicon ML inference)
- scikit-learn, MCP (optional — MCP server)

## Setup

### Prerequisites

- Python 3.12+
- PostgreSQL
  - pg_trgm
  - pgvector (optional — for [MCP server](#mcp-server-optional))
- Tesseract

### Configure Database

```bash
psql -f sql/schema.sql
```

Expects local trust authentication (no password).

### Configure Settings

```bash
cp config.yaml.example config.yaml
```

### Install Dependencies

```bash
uv sync                          # ingest only (core)
uv sync --extra web              # + Flask search UI
uv sync --extra mcp              # + MCP server
uv sync --extra web --extra mcp  # everything
```

## Usage

### Ingest

```bash
uv run tsa-ingest
```

1. Walk the screenshot directory
2. Pre-process with CLAHE (to enhance contrast of dark-mode screenshots)
3. Run Tesseract for OCR
4. Insert into Postgres

- Skips previously-ingested files to support incremental updates and graceful restart after Ctrl+C.
- Displays a `tqdm` progress bar.

### Search (optional)

```bash
uv run tsa-web
```

Runs a local webserver at `http://localhost:5000`. Pass `--debug` for Flask
debug mode and auto-reload.

**Fuzzy search modes:**
- **Word** (default) — full-text search with stemming. "running" matches "run". Supports boolean syntax.
- **Char** — character-level trigram matching. Catches typos and OCR errors.
- **None** — exact substring match (case-insensitive).
- **Semantic** — meaning-based search over MLX embeddings. Shown only when the `mcp` extra is installed.

All modes also match against VLM image descriptions when present, and results
can be restricted with the From/To capture-date filters.

**Sort options:**
- **Best** (default) — relevance × recency
- **Strongest** — relevance only
- **Newest** / **Oldest** — chronological

**Best** sorts by relevance weighted by recency. For example, with a 30-day half-life: a 30-day-old result scores half as much as an identical match from today, 60-day-old 1/3, 90-day-old 1/4, etc. Formula: `relevance_score × 1 / (age / half_life + 1)`. Half-life is set in `config.yaml`.

**Related tweets:** Click "related" on any result to find similar screenshots using MinHash/LSH (via [datasketch](https://github.com/ekzhu/datasketch)). OCR text is normalized, split into word 3-shingles, and hashed into 128-permutation MinHash signatures. An in-memory LSH index is built on server start for fast approximate nearest-neighbor lookup. Results are ranked by estimated Jaccard similarity.

### MCP Server (optional)

Exposes the archive to LLM chat models (e.g. Qwen, Claude) as MCP tools for semantic search, topic clustering, and discourse tracing.

#### Prerequisites

- Apple Silicon Mac (MLX required for embeddings and VLM)
- pgvector extension installed in PostgreSQL

#### Setup

Install with `uv sync --extra mcp` (see [Install Dependencies](#install-dependencies)).

#### Embedding & VLM Backfill

Embeddings and VLM descriptions run as standalone tools, not at MCP startup:

```bash
uv run tsa-embed      # embed OCR text (Qwen3-Embedding-0.6B, ~20/sec)
uv run tsa-describe   # VLM image descriptions (Qwen2.5-VL-7B, ~7.5 sec/image)
```

Both are resumable — they process rows where the target column is NULL.

#### Run

```bash
uv run tsa-mcp
```

Starts instantly with whatever is already in the database.

#### Tools

Fourteen tools organized into three tiers:

**Orient** — cheap, fast, no embeddings:
- **`now()`** — Current date and time (UTC and local). Resolves relative references like "last week."
- **`archive_range()`** — First and last dates in the archive.
- **`count_screenshots(after?, before?)`** — Count screenshots in a time window.
- **`tweet_activity(query?, keywords?, users?, after?, before?, granularity?)`** — Histogram of tweet counts over time (by day/week/month/year). Always includes max similarity when query is provided.

**Explore** — embedding-based, discover structure:
- **`search_tweets(query?, keywords?, limit?, offset?, min_score?, after?, before?, users?, sort?)`** — Semantic, keyword, or hybrid search. Semantic matches meaning; keywords filter by exact words via PostgreSQL tsquery. Use both together for precision.
- **`list_topics(after?, before?, users?, max_topics?)`** — Lightweight table of contents: topic label + tweet count, ranked by size. Uses PCA dimensionality reduction and HDBSCAN clustering. Supports user filtering.
- **`summarize_period(after?, before?, topics?, users?, max_topics?)`** — Rich clustered detail per topic: date span, tweet count, top mentioned users, and representative snippet. Supports topic and user filtering — e.g. "What was @someone talking about in March?" At least one of date range, topics, or users required.
- **`top_users(query?, after?, before?, limit?)`** — Who appears most in tweets about a topic. Embed the query, fetch relevant tweets, aggregate mentioned users by count.
- **`similar_users(handle, after?, before?, limit?, k?)`** — Who talks about similar things. Per-tweet nearest neighbors: for each tweet mentioning the handle, finds the K nearest tweets that don't mention that handle, aggregates users from those neighbors. Handles users with diverse interests without blurring.

**Drill** — follow threads once you have a foothold:
- **`get_tweet(id)`** — Full OCR text of a specific screenshot, plus the VLM transcription and the exact text of matched tweets from your Twitter export.
- **`find_related(id, limit?)`** — Lexically similar tweets via MinHash. Finds other parts of the same thread, conversation, or reply chain.
- **`nearby_screenshots(id, before?, after?)`** — Screenshots captured around the same time as a given tweet. Not a search — just chronological neighbors. Requires a known ID from another tool.
- **`search_by_user(handle, limit?, offset?, after?, before?, sort?)`** — Tweets mentioning a specific @user. Sort by "newest" (default) or "oldest".
- **`interactions(user1, user2, limit?, offset?, after?, before?)`** — Tweets where two users appear together — conversations, quote tweets, and reply chains.

#### Customization

Copy the example prompt file to provide personal context to the LLM:

```bash
cp mcp_prompt.txt.example mcp_prompt.txt
```

Edit `mcp_prompt.txt` with your interests, terminology, and preferences. This is injected into the MCP server instructions on startup — no need to edit system prompts per session. The file is gitignored.

#### MCP Client Configuration

```json
{
  "mcpServers": {
    "twitter-archive": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/twitter-screenshot-archive", "tsa-mcp"]
    }
  }
}
```

### Twitter Export Import (optional)

```bash
uv run tsa-import ~/path/to/twitter-export
uv run tsa-match
```

`tsa-import` loads likes and tweets from a Twitter data export (the
`window.YTD.*` JS files) into Postgres. `tsa-match` links screenshots to
exported tweets by word 3-gram seeding plus fuzzy scoring, then backfills
exact tweet timestamps from the matched tweets' snowflake IDs.

## Project Structure

```
src/twitter_screenshot_archive/
    core/           # Shared infrastructure: db, config, cleaning, ingest, minhash
    web/            # Flask search UI, templates, static assets
    mcp/            # MCP server, tools, clustering pipeline
```

Both `web/` and `mcp/` import from `core/`. Neither imports from the other. Each is independently installable via optional dependencies.
