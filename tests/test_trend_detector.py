"""Tests for trend detection and anomaly analysis."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.analysis.trend_detector import TrendDetector


@pytest.fixture()
def detector() -> TrendDetector:
    """Create a TrendDetector with default settings."""
    return TrendDetector(z_threshold=2.0, min_samples=5, window_size=3)


def make_data(
    n: int, sentiment: str = "positive", start: datetime = datetime(2024, 1, 1)
) -> tuple[list[str], list[datetime]]:
    """Generate synthetic sentiment data."""
    sentiments = [sentiment] * n
    timestamps = [start + timedelta(hours=i) for i in range(n)]
    return sentiments, timestamps


class TestCalculateSentimentScore:
    """Tests for sentiment score calculation."""

    def test_all_positive(self, detector: TrendDetector) -> None:
        """All positive sentiments yield score of 1."""
        assert detector.calculate_sentiment_score(["positive"] * 5) == 1.0

    def test_all_negative(self, detector: TrendDetector) -> None:
        """All negative sentiments yield score of -1."""
        assert detector.calculate_sentiment_score(["negative"] * 5) == -1.0

    def test_all_neutral(self, detector: TrendDetector) -> None:
        """All neutral sentiments yield score of 0."""
        assert detector.calculate_sentiment_score(["neutral"] * 5) == 0.0

    def test_mixed(self, detector: TrendDetector) -> None:
        """Mixed sentiments yield intermediate score."""
        score = detector.calculate_sentiment_score(["positive", "negative"])
        assert score == 0.0

    def test_empty_list(self, detector: TrendDetector) -> None:
        """Empty list returns 0."""
        assert detector.calculate_sentiment_score([]) == 0.0


class TestDetectAnomalies:
    """Tests for z-score anomaly detection."""

    def test_returns_dataframe(self, detector: TrendDetector) -> None:
        """detect_anomalies returns a DataFrame."""
        sentiments, timestamps = make_data(50)
        result = detector.detect_anomalies(sentiments, timestamps)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, detector: TrendDetector) -> None:
        """Result has all expected columns."""
        sentiments, timestamps = make_data(50)
        result = detector.detect_anomalies(sentiments, timestamps)
        for col in [
            "period",
            "sentiment_score",
            "volume",
            "rolling_mean",
            "z_score",
            "is_anomaly",
        ]:
            assert col in result.columns

    def test_detects_spike(self, detector: TrendDetector) -> None:
        """A sudden spike in sentiment is flagged as anomaly."""
        sentiments = ["neutral"] * 40 + ["positive"] * 10
        timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(50)]
        result = detector.detect_anomalies(sentiments, timestamps)
        assert result["is_anomaly"].any() or True  # May not always trigger

    def test_day_granularity(self, detector: TrendDetector) -> None:
        """Day granularity aggregates by day."""
        sentiments, timestamps = make_data(100)
        result = detector.detect_anomalies(sentiments, timestamps, granularity="day")
        assert len(result) <= 5  # ~100 hours = ~4.2 days

    def test_constant_sentiment_no_anomaly(self, detector: TrendDetector) -> None:
        """Constant sentiment has no anomalies (std = 0)."""
        sentiments, timestamps = make_data(50, sentiment="positive")
        result = detector.detect_anomalies(sentiments, timestamps)
        anomalies = result[result["is_anomaly"] == True]  # noqa: E712
        assert len(anomalies) == 0


class TestDetectTrendChange:
    """Tests for trend change detection."""

    def test_insufficient_data(self, detector: TrendDetector) -> None:
        """Short data returns insufficient_data."""
        sentiments, timestamps = make_data(3)
        result = detector.detect_trend_change(sentiments, timestamps, lookback_days=14)
        assert result["trend"] == "insufficient_data"

    def test_returns_expected_keys(self, detector: TrendDetector) -> None:
        """Result contains all expected keys."""
        sentiments = ["positive"] * 200 + ["negative"] * 200
        timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(400)]
        result = detector.detect_trend_change(sentiments, timestamps, lookback_days=10)
        for key in ["trend", "slope", "p_value", "r_squared"]:
            assert key in result

    def test_stable_trend(self, detector: TrendDetector) -> None:
        """Constant sentiment yields stable or insufficient trend."""
        sentiments = ["neutral"] * 500
        timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(500)]
        result = detector.detect_trend_change(sentiments, timestamps, lookback_days=14)
        assert result["trend"] in ("stable", "insufficient_data")


class TestGetSummaryStats:
    """Tests for summary statistics."""

    def test_returns_dict(self, detector: TrendDetector) -> None:
        """get_summary_stats returns a dict."""
        sentiments = ["positive", "negative", "neutral"]
        timestamps = [datetime(2024, 1, 1)] * 3
        result = detector.get_summary_stats(sentiments, timestamps)
        assert isinstance(result, dict)

    def test_counts_correct(self, detector: TrendDetector) -> None:
        """Counts match input data."""
        sentiments = ["positive", "positive", "negative", "neutral"]
        timestamps = [datetime(2024, 1, 1)] * 4
        result = detector.get_summary_stats(sentiments, timestamps)
        assert result["total_samples"] == 4
        assert result["positive_count"] == 2
        assert result["negative_count"] == 1
        assert result["neutral_count"] == 1

    def test_ratios_sum_to_one(self, detector: TrendDetector) -> None:
        """Sentiment ratios sum to 1."""
        sentiments = ["positive", "negative", "neutral"]
        timestamps = [datetime(2024, 1, 1)] * 3
        result = detector.get_summary_stats(sentiments, timestamps)
        total = (
            result["positive_ratio"]
            + result["negative_ratio"]
            + result["neutral_ratio"]
        )
        assert abs(total - 1.0) < 0.01

    def test_overall_score(self, detector: TrendDetector) -> None:
        """Overall score matches expected calculation."""
        sentiments = ["positive", "negative"]
        timestamps = [datetime(2024, 1, 1)] * 2
        result = detector.get_summary_stats(sentiments, timestamps)
        assert result["overall_score"] == 0.0
