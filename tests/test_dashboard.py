"""Tests for the Dash dashboard app and data loader."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import plotly.graph_objects as go

from src.dashboard.data_loader import load_benchmark_results, load_sentiment_data


class TestLoadSentimentData:
    """Tests for the data loader function."""

    def test_returns_dataframe(self, tmp_path: Path) -> None:
        """load_sentiment_data returns a DataFrame."""
        from src.data.database import init_database

        db_path = str(tmp_path / "test.duckdb")
        conn = init_database(db_path)
        conn.execute(
            "INSERT INTO tweets (id, text, processed_text, sentiment, source) "
            "VALUES ('1', 'test', 'test', 'positive', 'test')"
        )
        conn.close()

        df = load_sentiment_data(db_path=db_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_empty_database(self, tmp_path: Path) -> None:
        """Empty database returns empty DataFrame."""
        from src.data.database import init_database

        db_path = str(tmp_path / "empty.duckdb")
        init_database(db_path)

        df = load_sentiment_data(db_path=db_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_nonexistent_db_returns_empty(self) -> None:
        """Nonexistent database returns empty DataFrame."""
        df = load_sentiment_data(db_path="/nonexistent/path.duckdb")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestLoadBenchmarkResults:
    """Tests for benchmark results loader."""

    def test_returns_dict(self) -> None:
        """load_benchmark_results returns a dict with expected keys."""
        result = load_benchmark_results()
        assert isinstance(result, dict)
        assert "comparison_df" in result
        assert "agreement_matrix" in result


class TestDashApp:
    """Tests for the Dash application setup."""

    def test_app_creation(self) -> None:
        """Dashboard app can be created without errors."""
        from src.dashboard.app import create_app

        app = create_app()
        assert app is not None

    def test_app_has_layout(self) -> None:
        """App has a layout set."""
        from src.dashboard.app import create_app

        app = create_app()
        assert app.layout is not None


class TestOverviewCallbacks:
    """Tests for overview page callback logic."""

    def test_update_overview_empty_data(self) -> None:
        """Callback handles empty data gracefully."""
        from src.dashboard.pages.overview import update_overview

        with patch(
            "src.dashboard.data_loader.load_sentiment_data",
            return_value=pd.DataFrame(),
        ):
            result = update_overview(0, None, None)
            assert result[0] == "N/A"
            assert result[1] == "No data"

    def test_update_overview_with_data(self) -> None:
        """Callback computes correct metrics from data."""
        from src.dashboard.pages.overview import update_overview

        mock_df = pd.DataFrame(
            {
                "sentiment": ["positive", "positive", "negative"],
                "created_at": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03"]
                ),
                "text": ["a", "b", "c"],
            }
        )
        with patch(
            "src.dashboard.data_loader.load_sentiment_data",
            return_value=mock_df,
        ):
            result = update_overview(0, None, None)
            assert result[2] == "3"
            assert isinstance(result[4], go.Figure)
            assert isinstance(result[5], go.Figure)


class TestTrendsCallbacks:
    """Tests for trends page callback logic."""

    def test_update_trends_empty_data(self) -> None:
        """Callback handles empty data gracefully."""
        from src.dashboard.pages.trends import update_trends

        with patch(
            "src.dashboard.data_loader.load_sentiment_data",
            return_value=pd.DataFrame(),
        ):
            fig1, fig2 = update_trends(0, 7)
            assert isinstance(fig1, go.Figure)
            assert isinstance(fig2, go.Figure)

    def test_update_trends_with_data(self) -> None:
        """Callback produces valid figures from data."""
        from src.dashboard.pages.trends import update_trends

        mock_df = pd.DataFrame(
            {
                "sentiment": ["positive"] * 30 + ["negative"] * 20,
                "created_at": pd.date_range("2024-01-01", periods=50, freq="h"),
            }
        )
        with patch(
            "src.dashboard.data_loader.load_sentiment_data",
            return_value=mock_df,
        ):
            fig1, fig2 = update_trends(0, 7)
            assert isinstance(fig1, go.Figure)
            assert isinstance(fig2, go.Figure)
