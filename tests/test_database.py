"""Tests for DuckDB database initialization."""

from pathlib import Path

import duckdb
import pytest

from src.data.database import get_connection, init_database


class TestInitDatabase:
    """Tests for the init_database function."""

    def test_creates_database_file(self, tmp_path: Path) -> None:
        """Database file is created at the specified path."""
        db_path = str(tmp_path / "test.duckdb")
        conn = init_database(db_path)
        assert Path(db_path).exists()
        conn.close()

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """Parent directories are created if they don't exist."""
        db_path = str(tmp_path / "nested" / "dir" / "test.duckdb")
        conn = init_database(db_path)
        assert Path(db_path).exists()
        conn.close()

    def test_tweets_table_exists(self, tmp_path: Path) -> None:
        """The tweets table is created with the correct schema."""
        db_path = str(tmp_path / "test.duckdb")
        conn = init_database(db_path)
        result = conn.execute("SELECT * FROM tweets LIMIT 0").description
        column_names = [col[0] for col in result]
        expected = [
            "id",
            "text",
            "processed_text",
            "created_at",
            "sentiment",
            "sentiment_score",
            "confidence",
            "topic_id",
            "entities",
            "source",
        ]
        assert column_names == expected
        conn.close()

    def test_indexes_created(self, tmp_path: Path) -> None:
        """Indexes are created on the tweets table."""
        db_path = str(tmp_path / "test.duckdb")
        conn = init_database(db_path)
        indexes = conn.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'tweets'"
        ).fetchall()
        index_names = [idx[0] for idx in indexes]
        assert "idx_date" in index_names
        assert "idx_sentiment" in index_names
        assert "idx_topic" in index_names
        conn.close()

    def test_idempotent_init(self, tmp_path: Path) -> None:
        """Calling init_database twice doesn't raise an error."""
        db_path = str(tmp_path / "test.duckdb")
        conn1 = init_database(db_path)
        conn1.close()
        conn2 = init_database(db_path)
        result = conn2.execute("SELECT count(*) FROM tweets").fetchone()
        assert result[0] == 0
        conn2.close()

    def test_insert_and_query(self, tmp_path: Path) -> None:
        """Data can be inserted and queried from the tweets table."""
        db_path = str(tmp_path / "test.duckdb")
        conn = init_database(db_path)
        conn.execute("""
            INSERT INTO tweets (id, text, processed_text, sentiment, source)
            VALUES ('test_1', 'hello world', 'hello world', 'positive', 'test')
        """)
        result = conn.execute("SELECT count(*) FROM tweets").fetchone()
        assert result[0] == 1
        conn.close()


class TestGetConnection:
    """Tests for the get_connection function."""

    def test_returns_connection(self, tmp_path: Path) -> None:
        """get_connection returns a valid DuckDB connection."""
        db_path = str(tmp_path / "test.duckdb")
        init_database(db_path)
        conn = get_connection(db_path)
        assert conn is not None
        conn.close()

    def test_read_only_connection(self, tmp_path: Path) -> None:
        """Read-only connection prevents writes."""
        db_path = str(tmp_path / "test.duckdb")
        init_database(db_path)
        conn = get_connection(db_path, read_only=True)
        with pytest.raises(duckdb.InvalidInputException):
            conn.execute(
                "INSERT INTO tweets (id, text, processed_text) VALUES ('x', 'y', 'z')"
            )
        conn.close()
