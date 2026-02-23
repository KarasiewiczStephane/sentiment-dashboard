"""Tests for BERTopic topic modeler."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.topic_modeler import TopicModeler


@pytest.fixture()
def mock_topic_modeler() -> TopicModeler:
    """Create a TopicModeler with mocked BERTopic internals."""
    with patch("src.models.topic_modeler.SentenceTransformer"):
        with patch("src.models.topic_modeler.BERTopic") as mock_bertopic_cls:
            mock_instance = MagicMock()
            mock_instance.get_topics.return_value = {
                -1: [("noise", 0.1)],
                0: [
                    ("happy", 0.5),
                    ("joy", 0.4),
                    ("great", 0.3),
                    ("love", 0.2),
                    ("good", 0.1),
                ],
                1: [
                    ("bad", 0.5),
                    ("sad", 0.4),
                    ("terrible", 0.3),
                    ("hate", 0.2),
                    ("awful", 0.1),
                ],
            }
            mock_instance.get_topic.side_effect = lambda tid: {
                -1: [("noise", 0.1)],
                0: [
                    ("happy", 0.5),
                    ("joy", 0.4),
                    ("great", 0.3),
                    ("love", 0.2),
                    ("good", 0.1),
                ],
                1: [
                    ("bad", 0.5),
                    ("sad", 0.4),
                    ("terrible", 0.3),
                    ("hate", 0.2),
                    ("awful", 0.1),
                ],
            }[tid]
            mock_instance.get_topic_info.return_value = pd.DataFrame(
                {
                    "Topic": [-1, 0, 1],
                    "Count": [5, 20, 15],
                    "Name": ["Outlier", "Topic 0", "Topic 1"],
                }
            )
            mock_instance.fit_transform.return_value = (
                [0, 1, 0, 1, 0],
                np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8], [0.7, 0.3]]),
            )
            mock_instance.transform.return_value = (
                [0, 1, 0, 1, 0],
                np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8], [0.7, 0.3]]),
            )
            mock_bertopic_cls.return_value = mock_instance

            modeler = TopicModeler(min_topic_size=2)
            return modeler


class TestTopicModeler:
    """Tests for the TopicModeler class."""

    def test_fit_sets_topics(self, mock_topic_modeler: TopicModeler) -> None:
        """fit() populates topics and topic_info."""
        texts = ["happy day", "sad day", "great news", "bad news", "good times"]
        mock_topic_modeler.fit(texts)
        assert mock_topic_modeler.topics is not None
        assert mock_topic_modeler.topic_info is not None

    def test_transform_returns_topics(self, mock_topic_modeler: TopicModeler) -> None:
        """transform() returns topic assignments and probabilities."""
        texts = ["test text"] * 5
        topics, probs = mock_topic_modeler.transform(texts)
        assert len(topics) == 5
        assert probs.shape[0] == 5

    def test_get_topic_labels(self, mock_topic_modeler: TopicModeler) -> None:
        """get_topic_labels returns readable labels for each topic."""
        labels = mock_topic_modeler.get_topic_labels()
        assert -1 in labels
        assert labels[-1] == "Outlier/Noise"
        assert 0 in labels
        assert "happy" in labels[0]

    def test_get_topic_distribution(self, mock_topic_modeler: TopicModeler) -> None:
        """get_topic_distribution returns a DataFrame with expected columns."""
        mock_topic_modeler.fit(["test"] * 5)
        dist = mock_topic_modeler.get_topic_distribution()
        assert isinstance(dist, pd.DataFrame)
        assert "Topic" in dist.columns
        assert "Count" in dist.columns
        assert "label" in dist.columns

    def test_track_evolution(self, mock_topic_modeler: TopicModeler) -> None:
        """track_evolution returns a DataFrame of topic counts per window."""
        texts = ["text"] * 5
        base = datetime(2024, 1, 1)
        timestamps = [base + timedelta(days=i) for i in range(5)]
        evolution = mock_topic_modeler.track_evolution(texts, timestamps, window_days=3)
        assert isinstance(evolution, pd.DataFrame)
        assert "window_start" in evolution.columns
        assert "total_count" in evolution.columns

    def test_get_topic_sentiment(self, mock_topic_modeler: TopicModeler) -> None:
        """get_topic_sentiment aggregates sentiment by topic."""
        texts = ["text"] * 5
        sentiments = ["positive", "negative", "positive", "negative", "neutral"]
        result = mock_topic_modeler.get_topic_sentiment(texts, sentiments)
        assert isinstance(result, pd.DataFrame)
        assert "dominant_sentiment" in result.columns
        assert "total" in result.columns

    def test_topic_sentiment_has_all_columns(
        self, mock_topic_modeler: TopicModeler
    ) -> None:
        """Topic sentiment includes positive, negative, and neutral columns."""
        texts = ["text"] * 5
        sentiments = ["positive"] * 5
        result = mock_topic_modeler.get_topic_sentiment(texts, sentiments)
        for col in ["positive", "negative", "neutral"]:
            assert col in result.columns


class TestTopicModelerLabels:
    """Tests for topic label generation."""

    def test_outlier_label(self, mock_topic_modeler: TopicModeler) -> None:
        """Topic -1 gets the Outlier/Noise label."""
        labels = mock_topic_modeler.get_topic_labels()
        assert labels[-1] == "Outlier/Noise"

    def test_label_word_count(self, mock_topic_modeler: TopicModeler) -> None:
        """Topic labels contain up to n_representative_words words."""
        labels = mock_topic_modeler.get_topic_labels()
        for tid, label in labels.items():
            if tid != -1:
                words = label.split(", ")
                assert len(words) <= mock_topic_modeler.n_representative_words
