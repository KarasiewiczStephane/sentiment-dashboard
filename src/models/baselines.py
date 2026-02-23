"""Baseline sentiment analysis models using VADER and TextBlob."""

import logging
from typing import Protocol

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


class SentimentAnalyzer(Protocol):
    """Protocol for sentiment analysis models."""

    def predict(self, texts: list[str]) -> list[dict]: ...


class VADERSentiment:
    """Rule-based sentiment analysis using VADER.

    VADER is particularly suited for social media text with its handling
    of slang, emoticons, and emphasis markers.
    """

    def __init__(self) -> None:
        self.analyzer = SentimentIntensityAnalyzer()

    def _compound_to_label(self, compound: float) -> str:
        """Convert VADER compound score to a sentiment label.

        Args:
            compound: VADER compound score in [-1, 1].

        Returns:
            Sentiment label string.
        """
        if compound >= 0.05:
            return "positive"
        elif compound <= -0.05:
            return "negative"
        return "neutral"

    def predict(self, texts: list[str]) -> list[dict]:
        """Predict sentiment for a list of texts.

        Args:
            texts: List of input texts.

        Returns:
            List of dicts with sentiment, confidence, and raw scores.
        """
        results = []
        for text in texts:
            scores = self.analyzer.polarity_scores(text)
            results.append(
                {
                    "sentiment": self._compound_to_label(scores["compound"]),
                    "confidence": abs(scores["compound"]),
                    "scores": scores,
                }
            )
        return results


class TextBlobSentiment:
    """Pattern-based sentiment analysis using TextBlob.

    Uses a naive Bayes and pattern-based approach for polarity
    and subjectivity estimation.
    """

    def _polarity_to_label(self, polarity: float) -> str:
        """Convert TextBlob polarity to a sentiment label.

        Args:
            polarity: Polarity score in [-1, 1].

        Returns:
            Sentiment label string.
        """
        if polarity > 0.1:
            return "positive"
        elif polarity < -0.1:
            return "negative"
        return "neutral"

    def predict(self, texts: list[str]) -> list[dict]:
        """Predict sentiment for a list of texts.

        Args:
            texts: List of input texts.

        Returns:
            List of dicts with sentiment, confidence, polarity, and subjectivity.
        """
        results = []
        for text in texts:
            blob = TextBlob(text)
            results.append(
                {
                    "sentiment": self._polarity_to_label(blob.sentiment.polarity),
                    "confidence": abs(blob.sentiment.polarity),
                    "polarity": blob.sentiment.polarity,
                    "subjectivity": blob.sentiment.subjectivity,
                }
            )
        return results
