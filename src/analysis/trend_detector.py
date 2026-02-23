"""Z-score based anomaly detection for sentiment trend analysis."""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class TrendDetector:
    """Detects significant sentiment shifts using statistical methods.

    Args:
        z_threshold: Z-score threshold for anomaly flagging.
        min_samples: Minimum samples required for analysis.
        window_size: Rolling window size in periods.
    """

    def __init__(
        self,
        z_threshold: float = 2.0,
        min_samples: int = 30,
        window_size: int = 7,
    ) -> None:
        self.z_threshold = z_threshold
        self.min_samples = min_samples
        self.window_size = window_size

    def calculate_sentiment_score(self, sentiments: list[str]) -> float:
        """Convert sentiment labels to a numeric score.

        Args:
            sentiments: List of sentiment label strings.

        Returns:
            Mean numeric score in [-1, 1].
        """
        mapping = {"negative": -1, "neutral": 0, "positive": 1}
        scores = [mapping.get(s, 0) for s in sentiments]
        return float(np.mean(scores)) if scores else 0.0

    def detect_anomalies(
        self,
        sentiments: list[str],
        timestamps: list[datetime],
        granularity: str = "hour",
    ) -> pd.DataFrame:
        """Detect sentiment anomalies using z-score analysis.

        Args:
            sentiments: List of sentiment labels.
            timestamps: Corresponding timestamps.
            granularity: Time aggregation level ('hour' or 'day').

        Returns:
            DataFrame with period scores, rolling stats, z-scores, and anomaly flags.
        """
        df = pd.DataFrame({"sentiment": sentiments, "timestamp": timestamps})

        freq = "h" if granularity == "hour" else "D"
        df["period"] = df["timestamp"].dt.floor(freq)

        period_scores = (
            df.groupby("period")
            .agg(
                sentiment_score=(
                    "sentiment",
                    lambda x: self.calculate_sentiment_score(x.tolist()),
                ),
                volume=("sentiment", "count"),
            )
            .reset_index()
        )

        min_periods = max(1, self.min_samples // self.window_size)
        period_scores["rolling_mean"] = (
            period_scores["sentiment_score"]
            .rolling(window=self.window_size, min_periods=min_periods)
            .mean()
        )
        period_scores["rolling_std"] = (
            period_scores["sentiment_score"]
            .rolling(window=self.window_size, min_periods=min_periods)
            .std()
        )

        period_scores["z_score"] = (
            period_scores["sentiment_score"] - period_scores["rolling_mean"]
        ) / period_scores["rolling_std"].replace(0, np.nan)

        period_scores["is_anomaly"] = period_scores["z_score"].abs() > self.z_threshold
        period_scores["anomaly_type"] = period_scores.apply(
            lambda x: (
                (
                    "positive_spike"
                    if x["z_score"] > self.z_threshold
                    else (
                        "negative_spike" if x["z_score"] < -self.z_threshold else None
                    )
                )
                if pd.notna(x["z_score"])
                else None
            ),
            axis=1,
        )

        return period_scores

    def detect_trend_change(
        self,
        sentiments: list[str],
        timestamps: list[datetime],
        lookback_days: int = 14,
    ) -> dict:
        """Detect significant trend changes using linear regression.

        Args:
            sentiments: List of sentiment labels.
            timestamps: Corresponding timestamps.
            lookback_days: Number of recent days to analyze.

        Returns:
            Dict with trend direction, slope, r-squared, p-value, and confidence.
        """
        df = pd.DataFrame({"sentiment": sentiments, "timestamp": timestamps})
        df["day"] = df["timestamp"].dt.floor("D")

        daily_scores = (
            df.groupby("day")
            .agg(
                score=(
                    "sentiment",
                    lambda x: self.calculate_sentiment_score(x.tolist()),
                )
            )
            .reset_index()
        )

        if len(daily_scores) < lookback_days:
            return {"trend": "insufficient_data", "slope": 0, "p_value": 1.0}

        recent = daily_scores.tail(lookback_days)
        x = np.arange(len(recent))
        y = recent["score"].values

        slope, _intercept, r_value, p_value, _std_err = stats.linregress(x, y)

        trend = "stable"
        if p_value < 0.05:
            if slope > 0.01:
                trend = "improving"
            elif slope < -0.01:
                trend = "declining"

        return {
            "trend": trend,
            "slope": float(slope),
            "r_squared": float(r_value**2),
            "p_value": float(p_value),
            "daily_change": float(slope),
            "confidence": float(1 - p_value),
        }

    def get_summary_stats(
        self,
        sentiments: list[str],
        timestamps: list[datetime],
    ) -> dict:
        """Calculate summary statistics for sentiment data.

        Args:
            sentiments: List of sentiment labels.
            timestamps: Corresponding timestamps.

        Returns:
            Dict with counts, ratios, overall score, and date range.
        """
        df = pd.DataFrame({"sentiment": sentiments, "timestamp": timestamps})
        counts = df["sentiment"].value_counts().to_dict()
        total = len(df)

        return {
            "total_samples": total,
            "positive_count": counts.get("positive", 0),
            "negative_count": counts.get("negative", 0),
            "neutral_count": counts.get("neutral", 0),
            "positive_ratio": counts.get("positive", 0) / total if total > 0 else 0,
            "negative_ratio": counts.get("negative", 0) / total if total > 0 else 0,
            "neutral_ratio": counts.get("neutral", 0) / total if total > 0 else 0,
            "overall_score": self.calculate_sentiment_score(sentiments),
            "date_range": (df["timestamp"].min(), df["timestamp"].max()),
        }
