"""Text preprocessing pipeline for tweet data."""

import logging
import re
from typing import Optional

import duckdb
import emoji
import pandas as pd
from langdetect import LangDetectException, detect

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Preprocesses tweet text with URL removal, mention handling, and validation.

    Args:
        min_length: Minimum text length after preprocessing.
        max_length: Maximum text length after preprocessing.
    """

    def __init__(self, min_length: int = 10, max_length: int = 280) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.mention_pattern = re.compile(r"@\w+")

    def remove_urls(self, text: str) -> str:
        """Replace URLs with a placeholder token.

        Args:
            text: Input text.

        Returns:
            Text with URLs replaced by [URL].
        """
        return self.url_pattern.sub("[URL]", text)

    def handle_mentions(self, text: str) -> str:
        """Replace @mentions with a placeholder token.

        Args:
            text: Input text.

        Returns:
            Text with mentions replaced by [USER].
        """
        return self.mention_pattern.sub("[USER]", text)

    def convert_emojis(self, text: str) -> str:
        """Convert emoji characters to their text descriptions.

        Args:
            text: Input text.

        Returns:
            Text with emojis converted to descriptive text.
        """
        return emoji.demojize(text, delimiters=(" ", " "))

    def is_english(self, text: str) -> bool:
        """Detect whether text is in English.

        Args:
            text: Input text.

        Returns:
            True if text is detected as English.
        """
        try:
            return detect(text) == "en"
        except LangDetectException:
            return False

    def validate_length(self, text: str) -> bool:
        """Check whether text length falls within acceptable bounds.

        Args:
            text: Input text.

        Returns:
            True if length is between min_length and max_length.
        """
        return self.min_length <= len(text) <= self.max_length

    def preprocess(self, text: str) -> Optional[str]:
        """Apply the full preprocessing pipeline to a single text.

        Steps: URL removal, mention handling, emoji conversion,
        lowercasing, whitespace normalization, length and language validation.

        Args:
            text: Raw input text.

        Returns:
            Preprocessed text, or None if validation fails.
        """
        text = self.remove_urls(text)
        text = self.handle_mentions(text)
        text = self.convert_emojis(text)
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)

        if not self.validate_length(text):
            return None
        if not self.is_english(text):
            return None

        return text


def preprocess_dataset(
    df: pd.DataFrame, preprocessor: TextPreprocessor
) -> pd.DataFrame:
    """Apply preprocessing to an entire DataFrame.

    Args:
        df: DataFrame with a 'text' column.
        preprocessor: TextPreprocessor instance to use.

    Returns:
        DataFrame with 'processed_text' column, rows failing validation dropped.
    """
    df = df.copy()
    df["processed_text"] = df["text"].apply(preprocessor.preprocess)
    df = df.dropna(subset=["processed_text"])
    logger.info("Preprocessed %d tweets (dropped %d)", len(df), len(df) - len(df))
    return df.reset_index(drop=True)


def store_tweets(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    source: str = "tweeteval",
) -> int:
    """Store preprocessed tweets in DuckDB.

    Args:
        conn: Active DuckDB connection.
        df: DataFrame with preprocessed tweet data.
        source: Data source identifier.

    Returns:
        Number of rows inserted.
    """
    insert_df = df.copy()
    insert_df["id"] = [f"{source}_{i}" for i in range(len(insert_df))]
    insert_df["source"] = source

    columns = [
        "id",
        "text",
        "processed_text",
        "created_at",
        "sentiment",
        "sentiment_score",
        "confidence",
        "topic_id",
        "entities",
        "source",
    ]

    for col in columns:
        if col not in insert_df.columns:
            insert_df[col] = None

    insert_df = insert_df[columns]

    conn.execute("INSERT INTO tweets SELECT * FROM insert_df")
    logger.info("Stored %d tweets from source '%s'", len(insert_df), source)
    return len(insert_df)
