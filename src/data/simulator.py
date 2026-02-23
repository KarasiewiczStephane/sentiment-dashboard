"""Real-time tweet feed simulator for dashboard testing."""

import logging
import random
import threading
import time
from datetime import datetime, timedelta
from queue import Empty, Queue
from typing import Callable, Generator, Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


class TweetSimulator:
    """Simulates a real-time tweet feed by emitting stored tweets at a configurable rate.

    Args:
        db_path: Path to the DuckDB database containing tweets.
        tweets_per_second: Rate at which tweets are emitted.
    """

    def __init__(self, db_path: str, tweets_per_second: float = 5.0) -> None:
        self.db_path = db_path
        self.tweets_per_second = tweets_per_second
        self.interval = 1.0 / tweets_per_second
        self._running = False
        self._queue: Queue = Queue()
        self._thread: Optional[threading.Thread] = None

    def _load_tweets(self) -> pd.DataFrame:
        """Load all tweets from the database.

        Returns:
            DataFrame of tweet records.
        """
        conn = duckdb.connect(self.db_path, read_only=True)
        df = conn.execute("SELECT * FROM tweets").fetchdf()
        conn.close()
        return df

    def stream(self) -> Generator[dict, None, None]:
        """Yield tweets one at a time with simulated timestamps.

        Yields:
            Dictionary containing tweet data with a simulated_at timestamp.
        """
        df = self._load_tweets()
        if df.empty:
            return

        tweets = df.to_dict("records")
        random.shuffle(tweets)

        base_time = datetime.now()
        for i, tweet in enumerate(tweets):
            if not self._running and self._thread is not None:
                break
            tweet["simulated_at"] = base_time + timedelta(seconds=i * self.interval)
            yield tweet
            time.sleep(self.interval)

    def start_background(self, callback: Optional[Callable] = None) -> None:
        """Start a background thread emitting tweets to the internal queue.

        Args:
            callback: Optional function called with each emitted tweet.
        """
        self._running = True

        def _run() -> None:
            for tweet in self.stream():
                if not self._running:
                    break
                self._queue.put(tweet)
                if callback:
                    callback(tweet)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        logger.info("Simulator started at %.1f tweets/sec", self.tweets_per_second)

    def stop(self) -> None:
        """Stop the background simulation thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Simulator stopped")

    def get_next(self, timeout: float = 1.0) -> Optional[dict]:
        """Get the next tweet from the queue.

        Args:
            timeout: Maximum wait time in seconds.

        Returns:
            Tweet dictionary or None if timeout elapsed.
        """
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def get_batch(self, size: int, timeout: float = 5.0) -> list[dict]:
        """Get a batch of tweets from the queue.

        Args:
            size: Number of tweets to collect.
            timeout: Maximum wait time in seconds.

        Returns:
            List of tweet dictionaries.
        """
        batch = []
        deadline = time.time() + timeout
        while len(batch) < size and time.time() < deadline:
            tweet = self.get_next(timeout=0.1)
            if tweet:
                batch.append(tweet)
        return batch


class ReplaySimulator(TweetSimulator):
    """Replays historical tweet data with time compression.

    Args:
        db_path: Path to the DuckDB database.
        tweets_per_second: Fallback rate when timestamps are missing.
    """

    def stream_with_time_compression(
        self, compression_factor: float = 60.0
    ) -> Generator[dict, None, None]:
        """Replay tweets preserving relative time gaps, compressed by a factor.

        A compression_factor of 60 replays 1 hour of data in 1 minute.

        Args:
            compression_factor: How much to compress time gaps.

        Yields:
            Tweet dictionaries in chronological order.
        """
        df = self._load_tweets()
        if df.empty:
            return

        df = df.sort_values("created_at")
        tweets = df.to_dict("records")

        if len(tweets) < 2:
            yield from tweets
            return

        for i in range(len(tweets) - 1):
            yield tweets[i]
            if tweets[i]["created_at"] and tweets[i + 1]["created_at"]:
                delta = (
                    tweets[i + 1]["created_at"] - tweets[i]["created_at"]
                ).total_seconds()
                time.sleep(max(0, delta / compression_factor))
        yield tweets[-1]
