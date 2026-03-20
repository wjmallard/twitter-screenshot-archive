CREATE DATABASE twitter_screenshot_archive;

\c twitter_screenshot_archive

CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE screenshots (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    file_path TEXT NOT NULL UNIQUE,
    ocr_text TEXT,
    ocr_text_clean TEXT,
    created_at TIMESTAMPTZ,
    created_at_local TIMESTAMP,
    timezone TEXT,
    width INT,
    height INT,
    file_size BIGINT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ocr_text_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', COALESCE(ocr_text, ''))) STORED,
    minhash_signature BYTEA,
    mentioned_users TEXT[],
    tweet_time TIMESTAMPTZ,
    tweet_time_source TEXT,
    embedding vector(1024),
    image_type TEXT,
    image_description TEXT
);

CREATE INDEX idx_screenshots_tsv ON screenshots USING GIN (ocr_text_tsv);
CREATE INDEX idx_screenshots_trgm ON screenshots USING GIN (ocr_text gin_trgm_ops);
CREATE INDEX idx_screenshots_created ON screenshots (created_at);
CREATE INDEX idx_screenshots_created_local ON screenshots (created_at_local);
CREATE INDEX idx_screenshots_mentioned_users ON screenshots USING GIN (mentioned_users);
CREATE INDEX idx_screenshots_tweet_time ON screenshots (tweet_time);
CREATE INDEX idx_screenshots_embedding ON screenshots USING hnsw (embedding vector_cosine_ops);

CREATE TABLE twitter_likes (
    tweet_id BIGINT PRIMARY KEY,
    full_text TEXT,
    expanded_url TEXT,
    account TEXT NOT NULL,
    snowflake_date TIMESTAMPTZ,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE twitter_tweets (
    tweet_id BIGINT PRIMARY KEY,
    full_text TEXT,
    created_at TIMESTAMPTZ,
    in_reply_to_tweet_id BIGINT,
    in_reply_to_user_id BIGINT,
    in_reply_to_screen_name TEXT,
    retweet_count INT,
    favorite_count INT,
    lang TEXT,
    account TEXT NOT NULL,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE screenshot_tweet_matches (
    screenshot_id BIGINT NOT NULL REFERENCES screenshots(id),
    tweet_id BIGINT NOT NULL,
    score REAL NOT NULL,
    matched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (screenshot_id, tweet_id)
);
