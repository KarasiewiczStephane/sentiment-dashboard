"""Generate a balanced sample dataset from TweetEval for testing."""

import pandas as pd
from datasets import load_dataset


def generate_sample(
    output_path: str = "data/sample/sample_tweets.csv", n: int = 999
) -> None:
    """Download and create a balanced sample CSV from TweetEval test split.

    Args:
        output_path: Path to write the CSV file.
        n: Total number of samples (balanced across 3 classes).
    """
    dataset = load_dataset("tweet_eval", "sentiment", split="test")
    df = pd.DataFrame(dataset)
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    df["sentiment"] = df["label"].map(label_map)

    per_class = n // 3
    sample = (
        df.groupby("sentiment")
        .apply(
            lambda x: x.sample(min(per_class, len(x)), random_state=42),
        )
        .reset_index(drop=True)
    )
    sample[["text", "sentiment"]].to_csv(output_path, index=False)
    print(f"Saved {len(sample)} samples to {output_path}")


if __name__ == "__main__":
    generate_sample()
