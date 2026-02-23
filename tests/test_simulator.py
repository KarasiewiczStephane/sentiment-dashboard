"""Tests for the real-time feed simulator."""

import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.data.database import init_database
from src.data.simulator import ReplaySimulator, TweetSimulator


@pytest.fixture()
def populated_db(tmp_path: Path) -> str:
    """Create a temporary database with sample tweets."""
    db_path = str(tmp_path / "test.duckdb")
    conn = init_database(db_path)
    for i in range(20):
        conn.execute(
            """
            INSERT INTO tweets (id, text, processed_text, sentiment, created_at, source)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                f"sim_{i}",
                f"Test tweet number {i}",
                f"test tweet number {i}",
                ["positive", "negative", "neutral"][i % 3],
                datetime(2024, 1, 1, 0, i),
                "test",
            ],
        )
    conn.close()
    return db_path


@pytest.fixture()
def empty_db(tmp_path: Path) -> str:
    """Create a temporary empty database."""
    db_path = str(tmp_path / "empty.duckdb")
    init_database(db_path)
    return db_path


class TestTweetSimulator:
    """Tests for TweetSimulator."""

    def test_stream_yields_tweets(self, populated_db: str) -> None:
        """stream() yields tweet dictionaries."""
        sim = TweetSimulator(populated_db, tweets_per_second=1000)
        tweets = list(sim.stream())
        assert len(tweets) == 20
        assert "simulated_at" in tweets[0]

    def test_stream_empty_db(self, empty_db: str) -> None:
        """stream() yields nothing for an empty database."""
        sim = TweetSimulator(empty_db, tweets_per_second=100)
        tweets = list(sim.stream())
        assert len(tweets) == 0

    def test_simulated_timestamps_increase(self, populated_db: str) -> None:
        """Simulated timestamps are monotonically increasing."""
        sim = TweetSimulator(populated_db, tweets_per_second=1000)
        tweets = list(sim.stream())
        for i in range(1, len(tweets)):
            assert tweets[i]["simulated_at"] > tweets[i - 1]["simulated_at"]

    def test_background_start_stop(self, populated_db: str) -> None:
        """Background thread can be started and stopped cleanly."""
        sim = TweetSimulator(populated_db, tweets_per_second=100)
        sim.start_background()
        assert sim._running is True
        time.sleep(0.2)
        sim.stop()
        assert sim._running is False

    def test_background_with_callback(self, populated_db: str) -> None:
        """Callback is invoked for each emitted tweet."""
        callback = MagicMock()
        sim = TweetSimulator(populated_db, tweets_per_second=100)
        sim.start_background(callback=callback)
        time.sleep(0.5)
        sim.stop()
        assert callback.call_count > 0

    def test_get_next(self, populated_db: str) -> None:
        """get_next retrieves a tweet from the queue."""
        sim = TweetSimulator(populated_db, tweets_per_second=100)
        sim.start_background()
        time.sleep(0.2)
        tweet = sim.get_next(timeout=1.0)
        sim.stop()
        assert tweet is not None
        assert "text" in tweet

    def test_get_next_timeout(self, empty_db: str) -> None:
        """get_next returns None on timeout with empty queue."""
        sim = TweetSimulator(empty_db, tweets_per_second=100)
        result = sim.get_next(timeout=0.1)
        assert result is None

    def test_get_batch(self, populated_db: str) -> None:
        """get_batch collects multiple tweets."""
        sim = TweetSimulator(populated_db, tweets_per_second=100)
        sim.start_background()
        time.sleep(0.3)
        batch = sim.get_batch(size=5, timeout=2.0)
        sim.stop()
        assert len(batch) >= 1

    def test_interval_calculation(self) -> None:
        """Interval is correctly calculated from tweets_per_second."""
        sim = TweetSimulator("dummy.db", tweets_per_second=10)
        assert sim.interval == pytest.approx(0.1)

    def test_stop_without_start(self, populated_db: str) -> None:
        """Stopping without starting does not raise."""
        sim = TweetSimulator(populated_db)
        sim.stop()


class TestReplaySimulator:
    """Tests for ReplaySimulator."""

    def test_replay_yields_tweets(self, populated_db: str) -> None:
        """Replay simulator yields tweets in order."""
        sim = ReplaySimulator(populated_db)
        tweets = []
        for tweet in sim.stream_with_time_compression(compression_factor=100000):
            tweets.append(tweet)
            if len(tweets) >= 5:
                break
        assert len(tweets) >= 1

    def test_replay_empty_db(self, empty_db: str) -> None:
        """Replay simulator handles empty database."""
        sim = ReplaySimulator(empty_db)
        tweets = list(sim.stream_with_time_compression(compression_factor=100000))
        assert len(tweets) == 0

    def test_single_tweet_replay(self, tmp_path: Path) -> None:
        """Replay with single tweet yields exactly one result."""
        db_path = str(tmp_path / "single.duckdb")
        conn = init_database(db_path)
        conn.execute(
            """
            INSERT INTO tweets (id, text, processed_text, sentiment, created_at, source)
            VALUES ('s1', 'only tweet', 'only tweet', 'positive', '2024-01-01', 'test')
            """
        )
        conn.close()
        sim = ReplaySimulator(db_path)
        tweets = list(sim.stream_with_time_compression(compression_factor=100000))
        assert len(tweets) == 1
