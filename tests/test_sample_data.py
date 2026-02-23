"""Tests for sample data integrity."""

from pathlib import Path

import pandas as pd
import pytest

SAMPLE_PATH = Path("data/sample/sample_tweets.csv")


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Load the sample dataset."""
    return pd.read_csv(SAMPLE_PATH)


class TestSampleData:
    """Tests for the sample tweets CSV."""

    def test_file_exists(self) -> None:
        """Sample CSV file exists on disk."""
        assert SAMPLE_PATH.exists()

    def test_has_expected_columns(self, sample_df: pd.DataFrame) -> None:
        """Sample CSV has text and sentiment columns."""
        assert "text" in sample_df.columns
        assert "sentiment" in sample_df.columns

    def test_row_count(self, sample_df: pd.DataFrame) -> None:
        """Sample CSV has expected number of rows."""
        assert len(sample_df) >= 900

    def test_sentiment_labels(self, sample_df: pd.DataFrame) -> None:
        """All sentiment labels are valid."""
        valid = {"positive", "negative", "neutral"}
        assert set(sample_df["sentiment"].unique()) == valid

    def test_balanced_classes(self, sample_df: pd.DataFrame) -> None:
        """Classes are approximately balanced."""
        counts = sample_df["sentiment"].value_counts()
        assert counts.min() >= counts.max() * 0.5

    def test_no_empty_text(self, sample_df: pd.DataFrame) -> None:
        """No empty text values."""
        assert sample_df["text"].notna().all()
        assert (sample_df["text"].str.len() > 0).all()
