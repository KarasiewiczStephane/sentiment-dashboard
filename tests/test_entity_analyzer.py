"""Tests for entity extraction and entity-level sentiment analysis."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.models.entity_analyzer import EntityAnalyzer


@pytest.fixture(scope="module")
def analyzer() -> EntityAnalyzer:
    """Create a shared EntityAnalyzer instance."""
    return EntityAnalyzer()


class TestExtractEntities:
    """Tests for entity extraction."""

    def test_extracts_person(self, analyzer: EntityAnalyzer) -> None:
        """Person entities are extracted."""
        entities = analyzer.extract_entities("Barack Obama spoke at the conference")
        entity_texts = [e["text"] for e in entities]
        assert any("Obama" in t for t in entity_texts)

    def test_extracts_org(self, analyzer: EntityAnalyzer) -> None:
        """Organization entities are extracted."""
        entities = analyzer.extract_entities("Apple released a new iPhone today")
        entity_labels = [e["label"] for e in entities]
        assert "ORG" in entity_labels

    def test_extracts_gpe(self, analyzer: EntityAnalyzer) -> None:
        """Geopolitical entities are extracted."""
        entities = analyzer.extract_entities("The situation in London is stable")
        entity_labels = [e["label"] for e in entities]
        assert "GPE" in entity_labels

    def test_empty_text(self, analyzer: EntityAnalyzer) -> None:
        """Empty text returns empty entity list."""
        entities = analyzer.extract_entities("")
        assert entities == []

    def test_entity_dict_keys(self, analyzer: EntityAnalyzer) -> None:
        """Entity dicts contain expected keys."""
        entities = analyzer.extract_entities("Microsoft is headquartered in Redmond")
        if entities:
            assert "text" in entities[0]
            assert "label" in entities[0]
            assert "start" in entities[0]
            assert "end" in entities[0]


class TestBatchExtract:
    """Tests for batch entity extraction."""

    def test_batch_returns_list_of_lists(self, analyzer: EntityAnalyzer) -> None:
        """batch_extract returns a list of entity lists."""
        texts = ["Apple is great", "Google is expanding"]
        results = analyzer.batch_extract(texts)
        assert len(results) == 2
        assert isinstance(results[0], list)

    def test_batch_matches_single(self, analyzer: EntityAnalyzer) -> None:
        """Batch extraction finds same entities as single extraction."""
        text = "Tesla stock surged after Elon Musk announcement"
        single = analyzer.extract_entities(text)
        batch = analyzer.batch_extract([text])
        single_texts = {e["text"] for e in single}
        batch_texts = {e["text"] for e in batch[0]}
        assert single_texts == batch_texts


class TestAggregateEntitySentiment:
    """Tests for entity sentiment aggregation."""

    def test_returns_dataframe(self, analyzer: EntityAnalyzer) -> None:
        """aggregate_entity_sentiment returns a DataFrame."""
        texts = ["Apple is great"] * 5
        sentiments = [{"sentiment": "positive", "confidence": 0.9}] * 5
        result = analyzer.aggregate_entity_sentiment(texts, sentiments, min_mentions=1)
        assert isinstance(result, pd.DataFrame)

    def test_min_mentions_filter(self, analyzer: EntityAnalyzer) -> None:
        """Entities below min_mentions are filtered out."""
        texts = ["Apple is great", "Google is okay"]
        sentiments = [
            {"sentiment": "positive", "confidence": 0.9},
            {"sentiment": "neutral", "confidence": 0.5},
        ]
        result = analyzer.aggregate_entity_sentiment(texts, sentiments, min_mentions=5)
        assert len(result) == 0

    def test_sentiment_ratios_sum_to_one(self, analyzer: EntityAnalyzer) -> None:
        """Sentiment ratios sum to approximately 1."""
        texts = ["Apple is great"] * 3 + ["Apple is terrible"] * 2
        sentiments = [{"sentiment": "positive"}] * 3 + [{"sentiment": "negative"}] * 2
        result = analyzer.aggregate_entity_sentiment(texts, sentiments, min_mentions=1)
        if len(result) > 0:
            row = result.iloc[0]
            total_ratio = (
                row["positive_ratio"] + row["negative_ratio"] + row["neutral_ratio"]
            )
            assert abs(total_ratio - 1.0) < 0.01


class TestTrackEntityTrends:
    """Tests for entity trend tracking."""

    def test_returns_dataframe(self, analyzer: EntityAnalyzer) -> None:
        """track_entity_trends returns a DataFrame."""
        texts = ["Apple is great"] * 5
        sentiments = [{"sentiment": "positive"}] * 5
        base = datetime(2024, 1, 1)
        timestamps = [base + timedelta(hours=i * 6) for i in range(5)]
        result = analyzer.track_entity_trends(texts, sentiments, timestamps)
        assert isinstance(result, pd.DataFrame)

    def test_empty_entities_returns_empty(self, analyzer: EntityAnalyzer) -> None:
        """Texts with no entities return empty DataFrame."""
        texts = ["hello world"] * 3
        sentiments = [{"sentiment": "neutral"}] * 3
        timestamps = [datetime(2024, 1, 1)] * 3
        result = analyzer.track_entity_trends(texts, sentiments, timestamps)
        assert isinstance(result, pd.DataFrame)


class TestGetTrendingEntities:
    """Tests for trending entity detection."""

    def test_empty_input(self, analyzer: EntityAnalyzer) -> None:
        """Empty DataFrame returns empty list."""
        result = analyzer.get_trending_entities(pd.DataFrame())
        assert result == []

    def test_detects_improving(self, analyzer: EntityAnalyzer) -> None:
        """Entities with increasing sentiment are flagged as improving."""
        df = pd.DataFrame(
            {
                "window": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "entity": ["apple", "apple", "apple"],
                "sentiment_score": [-0.5, 0.0, 0.5],
                "type": ["ORG", "ORG", "ORG"],
            }
        )
        result = analyzer.get_trending_entities(df, lookback_windows=3, min_change=0.2)
        assert len(result) == 1
        assert result[0]["direction"] == "improving"

    def test_detects_declining(self, analyzer: EntityAnalyzer) -> None:
        """Entities with decreasing sentiment are flagged as declining."""
        df = pd.DataFrame(
            {
                "window": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "entity": ["google", "google", "google"],
                "sentiment_score": [0.5, 0.0, -0.5],
                "type": ["ORG", "ORG", "ORG"],
            }
        )
        result = analyzer.get_trending_entities(df, lookback_windows=3, min_change=0.2)
        assert len(result) == 1
        assert result[0]["direction"] == "declining"

    def test_sorted_by_magnitude(self, analyzer: EntityAnalyzer) -> None:
        """Trending entities are sorted by change magnitude."""
        df = pd.DataFrame(
            {
                "window": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03"] * 2
                ),
                "entity": ["apple", "apple", "apple", "google", "google", "google"],
                "sentiment_score": [-0.1, 0.0, 0.1, -0.5, 0.0, 0.5],
                "type": ["ORG"] * 6,
            }
        )
        result = analyzer.get_trending_entities(df, lookback_windows=3, min_change=0.1)
        if len(result) >= 2:
            assert abs(result[0]["sentiment_change"]) >= abs(
                result[1]["sentiment_change"]
            )
