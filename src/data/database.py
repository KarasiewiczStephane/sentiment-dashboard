"""DuckDB database initialization and schema management."""

import logging
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)


def init_database(db_path: str = "data/sentiment.duckdb") -> duckdb.DuckDBPyConnection:
    """Initialize the DuckDB database with the sentiment analysis schema.

    Creates the tweets table and relevant indexes if they don't already exist.

    Args:
        db_path: File path for the DuckDB database.

    Returns:
        Active DuckDB connection.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(db_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS tweets (
            id VARCHAR PRIMARY KEY,
            text VARCHAR NOT NULL,
            processed_text VARCHAR NOT NULL,
            created_at TIMESTAMP,
            sentiment VARCHAR,
            sentiment_score DOUBLE,
            confidence DOUBLE,
            topic_id INTEGER,
            entities VARCHAR[],
            source VARCHAR DEFAULT 'tweeteval'
        )
    """)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON tweets(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sentiment ON tweets(sentiment)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_topic ON tweets(topic_id)")

    logger.info("Database initialized at %s", db_path)
    return conn


def get_connection(
    db_path: str = "data/sentiment.duckdb", read_only: bool = False
) -> duckdb.DuckDBPyConnection:
    """Get a DuckDB connection.

    Args:
        db_path: File path for the DuckDB database.
        read_only: Whether to open in read-only mode.

    Returns:
        DuckDB connection.
    """
    return duckdb.connect(db_path, read_only=read_only)
