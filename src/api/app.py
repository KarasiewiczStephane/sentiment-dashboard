"""FastAPI application for sentiment analysis REST API."""

import io
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    HealthResponse,
    SentimentResult,
    TextAnalyzeRequest,
    TopicInfo,
    TopicsResponse,
    TrendPoint,
    TrendResponse,
)

logger = logging.getLogger(__name__)

models: dict = {}


def _parse_period(period: str) -> tuple[timedelta, str]:
    """Parse a period string like '7d', '24h', '2w' into timedelta and granularity.

    Args:
        period: Period string with numeric value and unit suffix.

    Returns:
        Tuple of (timedelta, granularity).
    """
    value = int(period[:-1])
    unit = period[-1]
    if unit == "h":
        return timedelta(hours=value), "hour"
    if unit == "w":
        return timedelta(weeks=value), "day"
    return timedelta(days=value), "day"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load models on startup, clear on shutdown."""
    try:
        from src.models.roberta_classifier import RoBERTaSentimentClassifier

        classifier = RoBERTaSentimentClassifier()
        classifier.load("models/roberta_best.pt")
        models["roberta"] = classifier
        logger.info("RoBERTa model loaded")
    except Exception:
        logger.warning("Could not load RoBERTa model")

    try:
        from src.models.topic_modeler import TopicModeler

        models["topic_model"] = TopicModeler.load("models/bertopic")
        logger.info("Topic model loaded")
    except Exception:
        logger.warning("Could not load topic model")

    from src.analysis.trend_detector import TrendDetector

    models["trend_detector"] = TrendDetector()

    yield
    models.clear()


app = FastAPI(
    title="Sentiment Analysis API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: TextAnalyzeRequest) -> AnalyzeResponse:
    """Analyze sentiment of a single text.

    Args:
        request: Request containing the text to analyze.

    Returns:
        Sentiment analysis result.
    """
    classifier = models.get("roberta")
    if classifier is None:
        from src.models.baselines import VADERSentiment

        classifier = VADERSentiment()
        model_name = "vader"
    else:
        model_name = "roberta"

    result = classifier.predict([request.text])[0]
    return AnalyzeResponse(
        text=request.text,
        result=SentimentResult(
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            probabilities=result.get("probabilities"),
        ),
        model=model_name,
    )


@app.post("/analyze/batch", response_model=BatchAnalyzeResponse)
async def analyze_batch(request: BatchAnalyzeRequest) -> BatchAnalyzeResponse:
    """Analyze sentiment of multiple texts.

    Args:
        request: Request containing texts to analyze.

    Returns:
        Batch analysis results with timing.
    """
    classifier = models.get("roberta")
    if classifier is None:
        from src.models.baselines import VADERSentiment

        classifier = VADERSentiment()
        model_name = "vader"
    else:
        model_name = "roberta"

    start_time = time.time()
    results = classifier.predict(request.texts)
    processing_time = (time.time() - start_time) * 1000

    return BatchAnalyzeResponse(
        results=[
            AnalyzeResponse(
                text=text,
                result=SentimentResult(
                    sentiment=r["sentiment"],
                    confidence=r["confidence"],
                    probabilities=r.get("probabilities"),
                ),
                model=model_name,
            )
            for text, r in zip(request.texts, results)
        ],
        total=len(request.texts),
        processing_time_ms=processing_time,
    )


@app.get("/trends", response_model=TrendResponse)
async def get_trends(
    period: str = Query(default="7d", pattern=r"^\d+[dhw]$"),
) -> TrendResponse:
    """Get sentiment trends for the specified period.

    Args:
        period: Time period string (e.g. '7d', '24h', '2w').

    Returns:
        Trend data with anomaly flags.
    """
    from src.dashboard.data_loader import load_sentiment_data

    delta, granularity = _parse_period(period)
    end_date = datetime.now()
    start_date = end_date - delta

    df = load_sentiment_data(start_date, end_date)

    detector = models.get("trend_detector")
    if detector is None:
        from src.analysis.trend_detector import TrendDetector

        detector = TrendDetector()

    if df.empty:
        return TrendResponse(
            period=period, data=[], overall_trend="insufficient_data", avg_sentiment=0.0
        )

    anomalies = detector.detect_anomalies(
        df["sentiment"].tolist(), df["created_at"].tolist(), granularity
    )
    trend_info = detector.detect_trend_change(
        df["sentiment"].tolist(), df["created_at"].tolist()
    )

    data = [
        TrendPoint(
            timestamp=row["period"],
            sentiment_score=row["sentiment_score"],
            volume=row["volume"],
            is_anomaly=bool(row["is_anomaly"]),
        )
        for _, row in anomalies.iterrows()
    ]

    return TrendResponse(
        period=period,
        data=data,
        overall_trend=trend_info["trend"],
        avg_sentiment=float(anomalies["sentiment_score"].mean()),
    )


@app.get("/topics", response_model=TopicsResponse)
async def get_topics() -> TopicsResponse:
    """Get current topics with sentiment breakdown.

    Returns:
        Topic list with sentiment ratios.
    """
    from src.dashboard.data_loader import load_sentiment_data

    topic_model = models.get("topic_model")
    if topic_model is None:
        return TopicsResponse(topics=[], total_topics=0)

    df = load_sentiment_data()
    if df.empty or "processed_text" not in df.columns:
        return TopicsResponse(topics=[], total_topics=0)

    topic_sentiment = topic_model.get_topic_sentiment(
        df["processed_text"].tolist(), df["sentiment"].tolist()
    )
    labels = topic_model.get_topic_labels()

    topics = []
    for topic_id, row in topic_sentiment.iterrows():
        if topic_id == -1:
            continue
        total = row.get("total", 1)
        topics.append(
            TopicInfo(
                topic_id=int(topic_id),
                label=labels.get(topic_id, f"Topic {topic_id}"),
                count=int(total),
                dominant_sentiment=str(row.get("dominant_sentiment", "neutral")),
                positive_ratio=float(
                    row.get("positive", 0) / total if total > 0 else 0
                ),
                negative_ratio=float(
                    row.get("negative", 0) / total if total > 0 else 0
                ),
            )
        )

    return TopicsResponse(topics=topics, total_topics=len(topics))


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API and dependency health.

    Returns:
        Health status with model and database info.
    """
    db_connected = False
    try:
        import duckdb

        conn = duckdb.connect("data/sentiment.duckdb", read_only=True)
        conn.execute("SELECT 1")
        conn.close()
        db_connected = True
    except Exception:
        pass

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded={
            "roberta": "roberta" in models,
            "topic_model": "topic_model" in models,
        },
        database_connected=db_connected,
    )


@app.get("/export")
async def export_data(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    sentiment: Optional[str] = Query(None, pattern=r"^(positive|negative|neutral)$"),
    topic_id: Optional[int] = None,
) -> StreamingResponse:
    """Export filtered sentiment data as CSV.

    Args:
        start_date: Filter start date.
        end_date: Filter end date.
        sentiment: Filter by sentiment label.
        topic_id: Filter by topic ID.

    Returns:
        CSV file as streaming response.
    """
    from src.dashboard.data_loader import load_sentiment_data

    df = load_sentiment_data(start_date, end_date)

    if sentiment:
        df = df[df["sentiment"] == sentiment]
    if topic_id is not None and "topic_id" in df.columns:
        df = df[df["topic_id"] == topic_id]

    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sentiment_export.csv"},
    )
