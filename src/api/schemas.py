"""Pydantic request and response schemas for the sentiment analysis API."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TextAnalyzeRequest(BaseModel):
    """Request body for single-text sentiment analysis."""

    text: str = Field(..., min_length=1, max_length=1000)


class BatchAnalyzeRequest(BaseModel):
    """Request body for batch sentiment analysis."""

    texts: list[str] = Field(..., min_length=1, max_length=100)


class SentimentResult(BaseModel):
    """Sentiment prediction result."""

    sentiment: str
    confidence: float
    probabilities: Optional[dict[str, float]] = None


class AnalyzeResponse(BaseModel):
    """Response for single-text analysis."""

    text: str
    result: SentimentResult
    model: str = "roberta"


class BatchAnalyzeResponse(BaseModel):
    """Response for batch analysis."""

    results: list[AnalyzeResponse]
    total: int
    processing_time_ms: float


class TrendPoint(BaseModel):
    """Single data point in a trend series."""

    timestamp: datetime
    sentiment_score: float
    volume: int
    is_anomaly: bool = False


class TrendResponse(BaseModel):
    """Response for trend queries."""

    period: str
    data: list[TrendPoint]
    overall_trend: str
    avg_sentiment: float


class TopicInfo(BaseModel):
    """Single topic with sentiment breakdown."""

    topic_id: int
    label: str
    count: int
    dominant_sentiment: str
    positive_ratio: float
    negative_ratio: float


class TopicsResponse(BaseModel):
    """Response for topic listing."""

    topics: list[TopicInfo]
    total_topics: int


class HealthResponse(BaseModel):
    """Response for health check."""

    status: str
    version: str
    models_loaded: dict[str, bool]
    database_connected: bool
