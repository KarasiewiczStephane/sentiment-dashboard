"""Named entity recognition and entity-level sentiment analysis."""

import logging
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import spacy

logger = logging.getLogger(__name__)


class EntityAnalyzer:
    """Extracts named entities and tracks per-entity sentiment.

    Args:
        model_name: spaCy model name for NER.
    """

    ENTITY_TYPES: set[str] = {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT"}

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            import subprocess

            subprocess.run(
                ["python", "-m", "spacy", "download", model_name], check=True
            )
            self.nlp = spacy.load(model_name)
        logger.info("Loaded spaCy model: %s", model_name)

    def extract_entities(self, text: str) -> list[dict]:
        """Extract named entities from a single text.

        Args:
            text: Input text string.

        Returns:
            List of dicts with text, label, start, and end keys.
        """
        doc = self.nlp(text)
        return [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in doc.ents
            if ent.label_ in self.ENTITY_TYPES
        ]

    def batch_extract(self, texts: list[str]) -> list[list[dict]]:
        """Extract entities from multiple texts efficiently.

        Args:
            texts: List of input texts.

        Returns:
            List of entity lists, one per input text.
        """
        results = []
        for doc in self.nlp.pipe(texts, batch_size=50):
            entities = [
                {"text": ent.text, "label": ent.label_}
                for ent in doc.ents
                if ent.label_ in self.ENTITY_TYPES
            ]
            results.append(entities)
        return results

    def aggregate_entity_sentiment(
        self,
        texts: list[str],
        sentiments: list[dict],
        min_mentions: int = 3,
    ) -> pd.DataFrame:
        """Aggregate sentiment scores per entity across all texts.

        Args:
            texts: List of input texts.
            sentiments: List of dicts with 'sentiment' and optional 'confidence' keys.
            min_mentions: Minimum entity mentions required for inclusion.

        Returns:
            DataFrame with entity, type, mentions, sentiment ratios, and dominant sentiment.
        """
        entity_sentiments: dict[tuple, list] = defaultdict(list)

        all_entities = self.batch_extract(texts)
        for entities, sentiment in zip(all_entities, sentiments):
            for entity in entities:
                key = (entity["text"].lower(), entity["label"])
                entity_sentiments[key].append(
                    {
                        "sentiment": sentiment["sentiment"],
                        "confidence": sentiment.get("confidence", 1.0),
                    }
                )

        rows = []
        for (entity_text, entity_type), mentions in entity_sentiments.items():
            if len(mentions) >= min_mentions:
                sentiment_counts: dict[str, int] = defaultdict(int)
                confidences = []
                for m in mentions:
                    sentiment_counts[m["sentiment"]] += 1
                    confidences.append(m["confidence"])

                total = len(mentions)
                rows.append(
                    {
                        "entity": entity_text,
                        "type": entity_type,
                        "mentions": total,
                        "positive_ratio": sentiment_counts["positive"] / total,
                        "negative_ratio": sentiment_counts["negative"] / total,
                        "neutral_ratio": sentiment_counts["neutral"] / total,
                        "avg_confidence": float(np.mean(confidences)),
                        "dominant_sentiment": max(
                            sentiment_counts, key=sentiment_counts.get
                        ),
                    }
                )

        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values("mentions", ascending=False)
        return df

    def track_entity_trends(
        self,
        texts: list[str],
        sentiments: list[dict],
        timestamps: list[datetime],
        window_hours: int = 24,
    ) -> pd.DataFrame:
        """Track entity sentiment trends over time windows.

        Args:
            texts: List of input texts.
            sentiments: List of sentiment result dicts.
            timestamps: Corresponding timestamps.
            window_hours: Size of time windows in hours.

        Returns:
            DataFrame with entity sentiment scores per window.
        """
        all_entities = self.batch_extract(texts)

        data = []
        for entities, sentiment, ts in zip(all_entities, sentiments, timestamps):
            for entity in entities:
                data.append(
                    {
                        "entity": entity["text"].lower(),
                        "type": entity["label"],
                        "sentiment": sentiment["sentiment"],
                        "timestamp": ts,
                    }
                )

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["window"] = df["timestamp"].dt.floor(f"{window_hours}h")
        trends = (
            df.groupby(["window", "entity"])
            .agg(
                sentiment_score=(
                    "sentiment",
                    lambda x: (x == "positive").mean() - (x == "negative").mean(),
                ),
                type=("type", "first"),
            )
            .reset_index()
        )

        return trends

    def get_trending_entities(
        self,
        entity_df: pd.DataFrame,
        lookback_windows: int = 3,
        min_change: float = 0.2,
    ) -> list[dict]:
        """Identify entities with significant sentiment changes.

        Args:
            entity_df: DataFrame from track_entity_trends.
            lookback_windows: Number of recent windows to compare.
            min_change: Minimum sentiment change threshold.

        Returns:
            List of trending entity dicts sorted by change magnitude.
        """
        if entity_df.empty:
            return []

        trending = []
        for entity in entity_df["entity"].unique():
            entity_data = entity_df[entity_df["entity"] == entity].sort_values("window")
            if len(entity_data) >= lookback_windows:
                recent = entity_data.tail(lookback_windows)
                change = (
                    recent["sentiment_score"].iloc[-1]
                    - recent["sentiment_score"].iloc[0]
                )
                if abs(change) >= min_change:
                    trending.append(
                        {
                            "entity": entity,
                            "type": entity_data["type"].iloc[0],
                            "sentiment_change": change,
                            "direction": "improving" if change > 0 else "declining",
                            "current_score": recent["sentiment_score"].iloc[-1],
                        }
                    )

        return sorted(trending, key=lambda x: abs(x["sentiment_change"]), reverse=True)
