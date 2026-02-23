"""Tests for dataset download utilities."""

from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.downloader import download_tweeteval


class TestDownloadTweeteval:
    """Tests for the download_tweeteval function."""

    @patch("src.data.downloader.load_dataset")
    def test_returns_dataframe(self, mock_load: MagicMock) -> None:
        """download_tweeteval returns a pandas DataFrame."""
        mock_load.return_value = [
            {"text": "great day", "label": 2},
            {"text": "bad day", "label": 0},
            {"text": "okay day", "label": 1},
        ]
        df = download_tweeteval(split="train")
        assert isinstance(df, pd.DataFrame)
        assert "text" in df.columns
        assert "sentiment" in df.columns

    @patch("src.data.downloader.load_dataset")
    def test_label_mapping(self, mock_load: MagicMock) -> None:
        """Numeric labels are mapped to sentiment strings."""
        mock_load.return_value = [
            {"text": "happy", "label": 2},
            {"text": "sad", "label": 0},
            {"text": "meh", "label": 1},
        ]
        df = download_tweeteval(split="train")
        assert set(df["sentiment"].unique()) == {"positive", "negative", "neutral"}

    @patch("src.data.downloader.load_dataset")
    def test_sample_size(self, mock_load: MagicMock) -> None:
        """Sampling limits the number of returned rows."""
        mock_load.return_value = [
            {"text": f"tweet {i}", "label": i % 3} for i in range(100)
        ]
        df = download_tweeteval(split="train", sample_size=10)
        assert len(df) == 10

    @patch("src.data.downloader.load_dataset")
    def test_sample_size_larger_than_data(self, mock_load: MagicMock) -> None:
        """Sample size larger than dataset returns all rows."""
        mock_load.return_value = [
            {"text": "tweet", "label": 0},
            {"text": "tweet2", "label": 1},
        ]
        df = download_tweeteval(split="train", sample_size=1000)
        assert len(df) == 2

    @patch("src.data.downloader.load_dataset")
    def test_no_sample_returns_all(self, mock_load: MagicMock) -> None:
        """Without sample_size, all rows are returned."""
        mock_load.return_value = [
            {"text": f"tweet {i}", "label": i % 3} for i in range(50)
        ]
        df = download_tweeteval(split="train")
        assert len(df) == 50
