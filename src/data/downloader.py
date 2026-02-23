"""Dataset download utilities for sentiment analysis data."""

import logging
from typing import Optional

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)


def download_tweeteval(
    split: str = "train", sample_size: Optional[int] = None
) -> pd.DataFrame:
    """Download TweetEval sentiment dataset from Hugging Face.

    Args:
        split: Dataset split to download (train, validation, test).
        sample_size: Optional number of samples to return. If None, returns all.

    Returns:
        DataFrame with text, label, and sentiment columns.
    """
    logger.info("Downloading TweetEval sentiment dataset (split=%s)", split)
    dataset = load_dataset("tweet_eval", "sentiment", split=split)
    df = pd.DataFrame(dataset)

    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    df["sentiment"] = df["label"].map(label_map)

    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    logger.info("Downloaded %d samples", len(df))
    return df
