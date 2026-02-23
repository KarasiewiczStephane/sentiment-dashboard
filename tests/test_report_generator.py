"""Tests for the executive report generator."""

from datetime import datetime
from pathlib import Path

import pytest

from src.analysis.report_generator import ReportGenerator


@pytest.fixture()
def generator() -> ReportGenerator:
    """Create a ReportGenerator instance."""
    return ReportGenerator(template_dir="templates")


@pytest.fixture()
def sample_data() -> dict:
    """Sample sentiment data for report generation."""
    return {
        "total": 1000,
        "avg_score": 0.25,
        "positive_count": 400,
        "neutral_count": 350,
        "negative_count": 250,
        "positive_ratio": 0.4,
        "negative_ratio": 0.25,
    }


class TestReportGenerator:
    """Tests for the ReportGenerator class."""

    def test_generate_report_returns_html(
        self, generator: ReportGenerator, sample_data: dict
    ) -> None:
        """generate_report returns an HTML string."""
        html = generator.generate_report(
            sentiment_data=sample_data,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 7),
        )
        assert isinstance(html, str)
        assert "<html>" in html
        assert "Sentiment Analysis" in html

    def test_report_contains_metrics(
        self, generator: ReportGenerator, sample_data: dict
    ) -> None:
        """Report HTML contains the key metrics."""
        html = generator.generate_report(sentiment_data=sample_data)
        assert "1000" in html
        assert "0.25" in html

    def test_report_with_topics(
        self, generator: ReportGenerator, sample_data: dict
    ) -> None:
        """Report includes topic data when provided."""
        topic_data = {
            "positive_topics": [
                {"label": "Happy Topic", "positive_ratio": 0.8, "count": 100}
            ],
            "negative_topics": [
                {"label": "Sad Topic", "negative_ratio": 0.7, "count": 50}
            ],
        }
        html = generator.generate_report(
            sentiment_data=sample_data, topic_data=topic_data
        )
        assert "Happy Topic" in html
        assert "Sad Topic" in html

    def test_report_with_entities(
        self, generator: ReportGenerator, sample_data: dict
    ) -> None:
        """Report includes trending entity data when provided."""
        entity_data = {
            "trending": [
                {
                    "entity": "TestCorp",
                    "type": "ORG",
                    "direction": "improving",
                    "sentiment_change": 0.5,
                }
            ]
        }
        html = generator.generate_report(
            sentiment_data=sample_data, entity_data=entity_data
        )
        assert "TestCorp" in html

    def test_report_with_anomalies(
        self, generator: ReportGenerator, sample_data: dict
    ) -> None:
        """Report includes anomaly data when provided."""
        trend_data = {
            "summary": "Sentiment is stable",
            "anomalies": [
                {
                    "period": "2024-01-05",
                    "anomaly_type": "positive_spike",
                    "z_score": 2.5,
                }
            ],
        }
        html = generator.generate_report(
            sentiment_data=sample_data, trend_data=trend_data
        )
        assert "positive_spike" in html
        assert "2.50" in html

    def test_report_no_anomalies(
        self, generator: ReportGenerator, sample_data: dict
    ) -> None:
        """Report shows no anomalies message when none detected."""
        html = generator.generate_report(sentiment_data=sample_data)
        assert "No significant anomalies" in html

    def test_save_html(
        self, generator: ReportGenerator, sample_data: dict, tmp_path: Path
    ) -> None:
        """save_html writes a valid HTML file."""
        html = generator.generate_report(sentiment_data=sample_data)
        output = str(tmp_path / "report.html")
        result = generator.save_html(html, output)
        assert Path(result).exists()
        content = Path(result).read_text()
        assert "<html>" in content

    def test_save_html_creates_dirs(
        self, generator: ReportGenerator, sample_data: dict, tmp_path: Path
    ) -> None:
        """save_html creates parent directories if needed."""
        html = generator.generate_report(sentiment_data=sample_data)
        output = str(tmp_path / "nested" / "dir" / "report.html")
        result = generator.save_html(html, output)
        assert Path(result).exists()

    def test_report_date_format(
        self, generator: ReportGenerator, sample_data: dict
    ) -> None:
        """Report includes formatted dates."""
        html = generator.generate_report(
            sentiment_data=sample_data,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 7),
        )
        assert "2024-01-01" in html
        assert "2024-01-07" in html
