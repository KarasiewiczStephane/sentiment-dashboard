"""Tests for the FastAPI REST API endpoints."""

from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import app, models


@pytest.fixture()
def client() -> TestClient:
    """Create a test client with mocked models."""
    models.clear()
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _mock_models():
    """Ensure models dict is clean for each test."""
    models.clear()
    yield
    models.clear()


class _FakeClassifier:
    """Minimal classifier for testing."""

    def predict(self, texts: list[str]) -> list[dict]:
        return [
            {
                "sentiment": "positive",
                "confidence": 0.95,
                "probabilities": {"positive": 0.95, "neutral": 0.03, "negative": 0.02},
            }
            for _ in texts
        ]


class TestAnalyzeEndpoint:
    """Tests for POST /analyze."""

    def test_analyze_single_text(self, client: TestClient) -> None:
        """Analyze returns a sentiment result for valid text."""
        models["roberta"] = _FakeClassifier()
        response = client.post("/analyze", json={"text": "I love this product!"})
        assert response.status_code == 200
        data = response.json()
        assert data["result"]["sentiment"] == "positive"
        assert data["model"] == "roberta"

    def test_analyze_falls_back_to_vader(self, client: TestClient) -> None:
        """Analyze uses VADER when RoBERTa is unavailable."""
        response = client.post("/analyze", json={"text": "I love this so much!"})
        assert response.status_code == 200
        data = response.json()
        assert data["result"]["sentiment"] in {"positive", "neutral", "negative"}
        assert data["model"] == "vader"

    def test_analyze_empty_text_rejected(self, client: TestClient) -> None:
        """Analyze rejects empty text."""
        response = client.post("/analyze", json={"text": ""})
        assert response.status_code == 422

    def test_analyze_text_too_long(self, client: TestClient) -> None:
        """Analyze rejects text exceeding max length."""
        response = client.post("/analyze", json={"text": "x" * 1001})
        assert response.status_code == 422

    def test_analyze_response_has_confidence(self, client: TestClient) -> None:
        """Analyze response includes confidence score."""
        models["roberta"] = _FakeClassifier()
        response = client.post("/analyze", json={"text": "Great work!"})
        data = response.json()
        assert 0 <= data["result"]["confidence"] <= 1


class TestBatchAnalyzeEndpoint:
    """Tests for POST /analyze/batch."""

    def test_batch_analyze(self, client: TestClient) -> None:
        """Batch analyze returns results for all texts."""
        models["roberta"] = _FakeClassifier()
        texts = ["I love this", "I hate this", "It is okay"]
        response = client.post("/analyze/batch", json={"texts": texts})
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["results"]) == 3
        assert data["processing_time_ms"] >= 0

    def test_batch_empty_list_rejected(self, client: TestClient) -> None:
        """Batch analyze rejects empty list."""
        response = client.post("/analyze/batch", json={"texts": []})
        assert response.status_code == 422

    def test_batch_falls_back_to_vader(self, client: TestClient) -> None:
        """Batch analyze falls back to VADER without RoBERTa."""
        response = client.post("/analyze/batch", json={"texts": ["Hello world"]})
        assert response.status_code == 200
        assert response.json()["results"][0]["model"] == "vader"


class TestTrendsEndpoint:
    """Tests for GET /trends."""

    def test_trends_empty_data(self, client: TestClient) -> None:
        """Trends returns empty response for no data."""
        with patch(
            "src.dashboard.data_loader.load_sentiment_data",
            return_value=pd.DataFrame(),
        ):
            response = client.get("/trends?period=7d")
            assert response.status_code == 200
            data = response.json()
            assert data["overall_trend"] == "insufficient_data"
            assert data["data"] == []

    def test_trends_invalid_period(self, client: TestClient) -> None:
        """Trends rejects invalid period format."""
        response = client.get("/trends?period=abc")
        assert response.status_code == 422

    def test_trends_valid_periods(self, client: TestClient) -> None:
        """Trends accepts valid period formats."""
        with patch(
            "src.dashboard.data_loader.load_sentiment_data",
            return_value=pd.DataFrame(),
        ):
            for period in ["7d", "24h", "2w"]:
                response = client.get(f"/trends?period={period}")
                assert response.status_code == 200


class TestTopicsEndpoint:
    """Tests for GET /topics."""

    def test_topics_no_model(self, client: TestClient) -> None:
        """Topics returns empty when no model loaded."""
        response = client.get("/topics")
        assert response.status_code == 200
        data = response.json()
        assert data["total_topics"] == 0
        assert data["topics"] == []


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_status(self, client: TestClient) -> None:
        """Health endpoint returns status info."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "roberta" in data["models_loaded"]
        assert "topic_model" in data["models_loaded"]

    def test_health_models_not_loaded(self, client: TestClient) -> None:
        """Health shows models as not loaded when empty."""
        response = client.get("/health")
        data = response.json()
        assert data["models_loaded"]["roberta"] is False
        assert data["models_loaded"]["topic_model"] is False


class TestExportEndpoint:
    """Tests for GET /export."""

    def test_export_returns_csv(self, client: TestClient) -> None:
        """Export returns CSV data."""
        mock_df = pd.DataFrame(
            {"text": ["hello"], "sentiment": ["positive"], "confidence": [0.9]}
        )
        with patch(
            "src.dashboard.data_loader.load_sentiment_data",
            return_value=mock_df,
        ):
            response = client.get("/export")
            assert response.status_code == 200
            assert "text/csv" in response.headers["content-type"]
            assert "hello" in response.text

    def test_export_filters_sentiment(self, client: TestClient) -> None:
        """Export filters by sentiment parameter."""
        mock_df = pd.DataFrame(
            {
                "text": ["good", "bad"],
                "sentiment": ["positive", "negative"],
                "confidence": [0.9, 0.8],
            }
        )
        with patch(
            "src.dashboard.data_loader.load_sentiment_data",
            return_value=mock_df,
        ):
            response = client.get("/export?sentiment=positive")
            assert response.status_code == 200
            assert "good" in response.text
            assert "bad" not in response.text

    def test_export_empty_data(self, client: TestClient) -> None:
        """Export returns empty CSV for no data."""
        with patch(
            "src.dashboard.data_loader.load_sentiment_data",
            return_value=pd.DataFrame(),
        ):
            response = client.get("/export")
            assert response.status_code == 200
