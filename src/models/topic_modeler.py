"""BERTopic-based topic modeling and evolution tracking."""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class TopicModeler:
    """Topic discovery and tracking using BERTopic.

    Args:
        embedding_model: Sentence transformer model name.
        nr_topics: Target number of topics (None for automatic).
        min_topic_size: Minimum documents per topic cluster.
        n_representative_words: Number of words for topic labels.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        nr_topics: Optional[int] = None,
        min_topic_size: int = 10,
        n_representative_words: int = 5,
    ) -> None:
        self.n_representative_words = n_representative_words
        self.sentence_model = SentenceTransformer(embedding_model)
        self.topic_model = BERTopic(
            embedding_model=self.sentence_model,
            nr_topics=nr_topics,
            min_topic_size=min_topic_size,
            calculate_probabilities=True,
        )
        self.topics: Optional[list[int]] = None
        self.probs: Optional[np.ndarray] = None
        self.topic_info: Optional[pd.DataFrame] = None

    def fit(self, texts: list[str]) -> "TopicModeler":
        """Fit the topic model on a corpus of texts.

        Args:
            texts: List of document strings.

        Returns:
            Self for method chaining.
        """
        self.topics, self.probs = self.topic_model.fit_transform(texts)
        self.topic_info = self.topic_model.get_topic_info()
        logger.info("Fit topic model with %d topics", len(self.topic_info))
        return self

    def transform(self, texts: list[str]) -> tuple[list[int], np.ndarray]:
        """Assign topics to new texts.

        Args:
            texts: List of document strings.

        Returns:
            Tuple of (topic assignments, probability matrix).
        """
        return self.topic_model.transform(texts)

    def get_topic_labels(self) -> dict[int, str]:
        """Get human-readable labels from top representative words.

        Returns:
            Dict mapping topic IDs to comma-separated word labels.
        """
        labels = {}
        for topic_id in self.topic_model.get_topics():
            if topic_id == -1:
                labels[topic_id] = "Outlier/Noise"
            else:
                words = self.topic_model.get_topic(topic_id)[
                    : self.n_representative_words
                ]
                labels[topic_id] = ", ".join([w[0] for w in words])
        return labels

    def get_topic_distribution(self) -> pd.DataFrame:
        """Get topic distribution with labels and counts.

        Returns:
            DataFrame with Topic, Count, label, and Name columns.
        """
        labels = self.get_topic_labels()
        df = self.topic_info.copy()
        df["label"] = df["Topic"].map(labels)
        return df[["Topic", "Count", "label", "Name"]]

    def track_evolution(
        self,
        texts: list[str],
        timestamps: list[datetime],
        window_days: int = 7,
    ) -> pd.DataFrame:
        """Track topic evolution over time windows.

        Args:
            texts: List of document strings.
            timestamps: Corresponding timestamps for each document.
            window_days: Time window size in days.

        Returns:
            DataFrame with topic counts per time window.
        """
        topics, _ = self.transform(texts)
        df = pd.DataFrame({"text": texts, "topic": topics, "timestamp": timestamps})

        min_date = df["timestamp"].min()
        max_date = df["timestamp"].max()

        evolution = []
        current = min_date
        while current <= max_date:
            window_end = current + timedelta(days=window_days)
            window_df = df[
                (df["timestamp"] >= current) & (df["timestamp"] < window_end)
            ]
            if len(window_df) > 0:
                topic_counts = window_df["topic"].value_counts().to_dict()
                evolution.append(
                    {
                        "window_start": current,
                        "window_end": window_end,
                        "total_count": len(window_df),
                        **{f"topic_{t}": c for t, c in topic_counts.items()},
                    }
                )
            current = window_end

        return pd.DataFrame(evolution)

    def get_topic_sentiment(
        self, texts: list[str], sentiments: list[str]
    ) -> pd.DataFrame:
        """Aggregate sentiment breakdown by topic.

        Args:
            texts: List of document strings.
            sentiments: Corresponding sentiment labels.

        Returns:
            DataFrame with sentiment counts per topic.
        """
        topics, _ = self.transform(texts)
        df = pd.DataFrame({"topic": topics, "sentiment": sentiments})
        result = df.groupby("topic")["sentiment"].value_counts().unstack(fill_value=0)
        for col in ["negative", "neutral", "positive"]:
            if col not in result.columns:
                result[col] = 0
        result["total"] = result.sum(axis=1)
        result["dominant_sentiment"] = result[
            ["negative", "neutral", "positive"]
        ].idxmax(axis=1)
        return result

    def save(self, path: str) -> None:
        """Save the topic model to disk.

        Args:
            path: Directory path for the saved model.
        """
        self.topic_model.save(path)
        logger.info("Topic model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "TopicModeler":
        """Load a topic model from disk.

        Args:
            path: Directory path of the saved model.

        Returns:
            TopicModeler instance with loaded model.
        """
        instance = cls.__new__(cls)
        instance.topic_model = BERTopic.load(path)
        instance.topic_info = instance.topic_model.get_topic_info()
        instance.n_representative_words = 5
        instance.topics = None
        instance.probs = None
        logger.info("Topic model loaded from %s", path)
        return instance
