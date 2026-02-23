"""Tests for advanced dashboard pages."""

from unittest.mock import patch

import pandas as pd
import plotly.graph_objects as go
import pytest


@pytest.fixture(autouse=True, scope="module")
def _init_dash_app():
    """Initialize Dash app before page imports."""
    from src.dashboard.app import create_app

    create_app()


class TestTopicsPage:
    """Tests for the topics page callbacks."""

    def test_empty_data(self) -> None:
        """Topics page handles empty data."""
        from src.dashboard.pages.topics import update_topics

        with (
            patch(
                "src.dashboard.data_loader.load_sentiment_data",
                return_value=pd.DataFrame(),
            ),
            patch("src.dashboard.data_loader.load_topic_model", return_value=None),
        ):
            fig1, fig2, fig3 = update_topics(0)
            assert isinstance(fig1, go.Figure)
            assert isinstance(fig2, go.Figure)
            assert isinstance(fig3, go.Figure)

    def test_no_topic_model(self) -> None:
        """Topics page handles missing topic model."""
        from src.dashboard.pages.topics import update_topics

        mock_df = pd.DataFrame({"sentiment": ["positive"], "processed_text": ["test"]})
        with (
            patch(
                "src.dashboard.data_loader.load_sentiment_data",
                return_value=mock_df,
            ),
            patch("src.dashboard.data_loader.load_topic_model", return_value=None),
        ):
            fig1, fig2, fig3 = update_topics(0)
            assert isinstance(fig1, go.Figure)


class TestEntitiesPage:
    """Tests for the entities page callbacks."""

    def test_empty_data(self) -> None:
        """Entities page handles empty data."""
        from src.dashboard.pages.entities import update_entities

        with patch(
            "src.dashboard.data_loader.load_sentiment_data",
            return_value=pd.DataFrame(),
        ):
            fig, trending, cards = update_entities(0)
            assert isinstance(fig, go.Figure)
            assert isinstance(trending, list)
            assert isinstance(cards, list)

    def test_no_processed_text(self) -> None:
        """Entities page handles missing processed_text column."""
        from src.dashboard.pages.entities import update_entities

        mock_df = pd.DataFrame({"sentiment": ["positive"]})
        with patch(
            "src.dashboard.data_loader.load_sentiment_data",
            return_value=mock_df,
        ):
            fig, trending, cards = update_entities(0)
            assert isinstance(fig, go.Figure)


class TestComparisonPage:
    """Tests for the model comparison page callbacks."""

    def test_empty_results(self) -> None:
        """Comparison page handles empty benchmark results."""
        from src.dashboard.pages.comparison import update_comparison

        with patch(
            "src.dashboard.data_loader.load_benchmark_results",
            return_value={
                "comparison_df": pd.DataFrame(),
                "agreement_matrix": pd.DataFrame(),
            },
        ):
            table, heatmap, speed = update_comparison(0)
            assert isinstance(heatmap, go.Figure)
            assert isinstance(speed, go.Figure)

    def test_with_results(self) -> None:
        """Comparison page renders with valid benchmark data."""
        from src.dashboard.pages.comparison import update_comparison

        comparison_df = pd.DataFrame(
            {
                "Model": ["VADER", "TextBlob"],
                "Accuracy": ["0.650", "0.600"],
                "Samples/sec": ["1000.0", "500.0"],
            }
        )
        agreement_matrix = pd.DataFrame(
            [[1.0, 0.7], [0.7, 1.0]],
            index=["VADER", "TextBlob"],
            columns=["VADER", "TextBlob"],
        )
        with patch(
            "src.dashboard.data_loader.load_benchmark_results",
            return_value={
                "comparison_df": comparison_df,
                "agreement_matrix": agreement_matrix,
            },
        ):
            table, heatmap, speed = update_comparison(0)
            assert isinstance(heatmap, go.Figure)
            assert isinstance(speed, go.Figure)
