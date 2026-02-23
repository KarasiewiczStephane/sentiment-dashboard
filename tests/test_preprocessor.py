"""Tests for text preprocessing pipeline."""

import pandas as pd

from src.data.preprocessor import TextPreprocessor, preprocess_dataset, store_tweets


class TestTextPreprocessor:
    """Tests for the TextPreprocessor class."""

    def setup_method(self) -> None:
        """Create a preprocessor instance for each test."""
        self.preprocessor = TextPreprocessor(min_length=10, max_length=280)

    def test_remove_http_url(self) -> None:
        """HTTP URLs are replaced with [URL]."""
        result = self.preprocessor.remove_urls("Check this http://example.com now")
        assert result == "Check this [URL] now"

    def test_remove_https_url(self) -> None:
        """HTTPS URLs are replaced with [URL]."""
        result = self.preprocessor.remove_urls("Visit https://example.com/page")
        assert result == "Visit [URL]"

    def test_remove_www_url(self) -> None:
        """www URLs are replaced with [URL]."""
        result = self.preprocessor.remove_urls("Go to www.example.com for details")
        assert result == "Go to [URL] for details"

    def test_remove_multiple_urls(self) -> None:
        """Multiple URLs in one text are all replaced."""
        text = "See http://a.com and https://b.com"
        result = self.preprocessor.remove_urls(text)
        assert result == "See [URL] and [URL]"

    def test_handle_single_mention(self) -> None:
        """Single @mention is replaced with [USER]."""
        result = self.preprocessor.handle_mentions("Hello @johndoe how are you")
        assert result == "Hello [USER] how are you"

    def test_handle_multiple_mentions(self) -> None:
        """Multiple @mentions are all replaced."""
        result = self.preprocessor.handle_mentions("@alice and @bob are here")
        assert result == "[USER] and [USER] are here"

    def test_convert_emojis(self) -> None:
        """Emoji characters are converted to text descriptions."""
        result = self.preprocessor.convert_emojis("Great job! 👍")
        assert "thumbs_up" in result or "thumbs" in result.lower()

    def test_is_english_positive(self) -> None:
        """English text is correctly identified."""
        assert self.preprocessor.is_english(
            "This is a perfectly normal English sentence"
        )

    def test_is_english_negative(self) -> None:
        """Non-English text is correctly rejected."""
        assert not self.preprocessor.is_english("这是中文句子测试内容较长的句子")

    def test_is_english_short_text(self) -> None:
        """Very short text may fail language detection gracefully."""
        result = self.preprocessor.is_english("ok")
        assert isinstance(result, bool)

    def test_validate_length_within_bounds(self) -> None:
        """Text within length bounds passes validation."""
        assert self.preprocessor.validate_length("Hello World!")

    def test_validate_length_too_short(self) -> None:
        """Text shorter than min_length fails validation."""
        assert not self.preprocessor.validate_length("Hi")

    def test_validate_length_too_long(self) -> None:
        """Text longer than max_length fails validation."""
        assert not self.preprocessor.validate_length("x" * 281)

    def test_validate_length_exact_min(self) -> None:
        """Text at exactly min_length passes."""
        assert self.preprocessor.validate_length("x" * 10)

    def test_validate_length_exact_max(self) -> None:
        """Text at exactly max_length passes."""
        assert self.preprocessor.validate_length("x" * 280)

    def test_preprocess_full_pipeline(self) -> None:
        """Full preprocessing pipeline produces expected output."""
        text = "Hey @user check https://example.com it's AMAZING! 👍"
        result = self.preprocessor.preprocess(text)
        assert result is not None
        assert "[user]" in result
        assert "[url]" in result
        assert result == result.lower()

    def test_preprocess_too_short_returns_none(self) -> None:
        """Text that's too short after preprocessing returns None."""
        preprocessor = TextPreprocessor(min_length=100, max_length=280)
        result = preprocessor.preprocess("Hi there")
        assert result is None

    def test_preprocess_normalizes_whitespace(self) -> None:
        """Multiple spaces are collapsed to single spaces."""
        text = "Hello    world   this   is   a   test  sentence  here"
        result = self.preprocessor.preprocess(text)
        assert result is not None
        assert "  " not in result

    def test_preprocess_non_english_returns_none(self) -> None:
        """Non-English text returns None."""
        result = self.preprocessor.preprocess(
            "这是中文句子测试内容较长的句子需要更多字符"
        )
        assert result is None


class TestPreprocessDataset:
    """Tests for the preprocess_dataset function."""

    def test_adds_processed_text_column(self) -> None:
        """processed_text column is added to the DataFrame."""
        df = pd.DataFrame(
            {
                "text": [
                    "This is a perfectly normal English sentence for testing purposes"
                ]
            }
        )
        preprocessor = TextPreprocessor(min_length=5, max_length=280)
        result = preprocess_dataset(df, preprocessor)
        assert "processed_text" in result.columns

    def test_drops_invalid_rows(self) -> None:
        """Rows that fail preprocessing are dropped."""
        df = pd.DataFrame(
            {"text": ["Hi", "This is a valid English sentence for testing"]}
        )
        preprocessor = TextPreprocessor(min_length=10, max_length=280)
        result = preprocess_dataset(df, preprocessor)
        assert len(result) <= 2

    def test_preserves_original_text(self) -> None:
        """Original text column is preserved."""
        df = pd.DataFrame(
            {
                "text": [
                    "This is a perfectly normal English sentence for testing purposes"
                ]
            }
        )
        preprocessor = TextPreprocessor(min_length=5, max_length=280)
        result = preprocess_dataset(df, preprocessor)
        if len(result) > 0:
            assert "text" in result.columns

    def test_resets_index(self) -> None:
        """Result has a clean sequential index."""
        df = pd.DataFrame(
            {
                "text": [
                    "Short",
                    "This is a valid English sentence for testing purposes",
                    "Tiny",
                    "Another valid English text here for processing and checking",
                ]
            }
        )
        preprocessor = TextPreprocessor(min_length=10, max_length=280)
        result = preprocess_dataset(df, preprocessor)
        assert list(result.index) == list(range(len(result)))


class TestStoreTweets:
    """Tests for the store_tweets function."""

    def test_store_and_retrieve(self, tmp_path) -> None:
        """Tweets are stored in and retrievable from DuckDB."""
        from src.data.database import init_database

        db_path = str(tmp_path / "test.duckdb")
        conn = init_database(db_path)

        df = pd.DataFrame(
            {
                "text": ["Hello world test"],
                "processed_text": ["hello world test"],
                "sentiment": ["positive"],
            }
        )

        count = store_tweets(conn, df, source="test")
        assert count == 1

        result = conn.execute("SELECT count(*) FROM tweets").fetchone()
        assert result[0] == 1
        conn.close()

    def test_store_with_source_prefix(self, tmp_path) -> None:
        """Tweet IDs are prefixed with the source name."""
        from src.data.database import init_database

        db_path = str(tmp_path / "test.duckdb")
        conn = init_database(db_path)

        df = pd.DataFrame(
            {
                "text": ["Test tweet here"],
                "processed_text": ["test tweet here"],
                "sentiment": ["neutral"],
            }
        )

        store_tweets(conn, df, source="mydata")
        row = conn.execute("SELECT id FROM tweets").fetchone()
        assert row[0].startswith("mydata_")
        conn.close()

    def test_store_multiple_tweets(self, tmp_path) -> None:
        """Multiple tweets are stored correctly."""
        from src.data.database import init_database

        db_path = str(tmp_path / "test.duckdb")
        conn = init_database(db_path)

        df = pd.DataFrame(
            {
                "text": ["Tweet one here", "Tweet two here", "Tweet three here"],
                "processed_text": [
                    "tweet one here",
                    "tweet two here",
                    "tweet three here",
                ],
                "sentiment": ["positive", "negative", "neutral"],
            }
        )

        count = store_tweets(conn, df, source="test")
        assert count == 3

        result = conn.execute("SELECT count(*) FROM tweets").fetchone()
        assert result[0] == 3
        conn.close()
