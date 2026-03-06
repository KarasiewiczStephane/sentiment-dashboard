"""Data loading utilities for the Dash dashboard."""

import logging
from datetime import datetime
from functools import lru_cache
from typing import Optional

import duckdb
import pandas as pd

from src.utils.config import load_config

logger = logging.getLogger(__name__)


def load_sentiment_data(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load sentiment data from DuckDB with optional date filtering.

    Args:
        start_date: Optional start date filter.
        end_date: Optional end date filter.
        db_path: Database path override. Uses config default if None.

    Returns:
        DataFrame with tweet sentiment data.
    """
    if db_path is None:
        config = load_config()
        db_path = config.database["duckdb_path"]

    try:
        conn = duckdb.connect(db_path, read_only=True)
    except Exception:
        logger.warning("Could not connect to database at %s", db_path)
        return pd.DataFrame()

    query = "SELECT * FROM tweets"
    conditions = []

    if start_date:
        conditions.append(f"created_at >= '{start_date}'")
    if end_date:
        conditions.append(f"created_at <= '{end_date}'")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    try:
        df = conn.execute(query).fetchdf()
    except Exception:
        logger.warning("Failed to query sentiment data")
        df = pd.DataFrame()
    finally:
        conn.close()

    return df


@lru_cache(maxsize=1)
def load_topic_model(model_path: str = "models/bertopic") -> Optional[object]:
    """Load the BERTopic model from disk (cached after first load).

    Args:
        model_path: Path to the saved BERTopic model.

    Returns:
        TopicModeler instance or None if loading fails.
    """
    try:
        from src.models.topic_modeler import TopicModeler

        return TopicModeler.load(model_path)
    except Exception:
        logger.warning("Could not load topic model from %s", model_path)
        return None


def load_benchmark_results() -> dict:
    """Load benchmark comparison results.

    Returns:
        Dict with comparison_df and agreement_matrix keys.
    """
    from pathlib import Path

    bench_dir = Path("data/benchmarks")
    comparison_path = bench_dir / "comparison.csv"
    agreement_path = bench_dir / "agreement.csv"

    comparison_df = pd.DataFrame()
    agreement_matrix = pd.DataFrame()

    if comparison_path.exists():
        comparison_df = pd.read_csv(comparison_path)
    if agreement_path.exists():
        agreement_matrix = pd.read_csv(agreement_path, index_col=0)

    return {
        "comparison_df": comparison_df,
        "agreement_matrix": agreement_matrix,
    }
