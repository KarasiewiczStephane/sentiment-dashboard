"""Populate the DuckDB database with full demo data for all dashboard pages."""

import random
from datetime import datetime, timedelta

import duckdb


def main() -> None:
    db_path = "data/sentiment.duckdb"
    conn = duckdb.connect(db_path)

    df = conn.execute("SELECT * FROM tweets").fetchdf()
    print(f"Loaded {len(df)} tweets from database")

    # 1. Generate synthetic timestamps spread over last 30 days
    print("Adding timestamps...")
    now = datetime.now()
    start = now - timedelta(days=30)
    random.seed(42)
    timestamps = sorted(
        [
            start + timedelta(seconds=random.uniform(0, 30 * 86400))
            for _ in range(len(df))
        ]
    )
    df["created_at"] = timestamps

    # 2. Run VADER for sentiment_score and confidence
    print("Running VADER sentiment scoring...")
    from src.models.baselines import VADERSentiment

    vader = VADERSentiment()
    vader_results = vader.predict(df["processed_text"].tolist())
    df["sentiment_score"] = [r["scores"]["compound"] for r in vader_results]
    df["confidence"] = [r["confidence"] for r in vader_results]

    # 3. Run spaCy NER for entities
    print("Extracting entities with spaCy...")
    from src.models.entity_analyzer import EntityAnalyzer

    analyzer = EntityAnalyzer()
    all_entities = analyzer.batch_extract(df["processed_text"].tolist())
    df["entities"] = [
        [e["text"] for e in ents] if ents else None for ents in all_entities
    ]

    # 4. Fit BERTopic for topic_id
    print("Fitting BERTopic (this may take a minute)...")
    from src.models.topic_modeler import TopicModeler

    modeler = TopicModeler(min_topic_size=10)
    modeler.fit(df["processed_text"].tolist())
    df["topic_id"] = modeler.topics
    modeler.save("models/bertopic")
    print(f"  Found {len(modeler.topic_info)} topics")

    # 5. Rewrite the tweets table with full data
    print("Updating database...")
    conn.execute("DELETE FROM tweets")
    conn.execute("INSERT INTO tweets SELECT * FROM df")
    conn.close()

    # 6. Generate benchmark comparison data
    print("Running model benchmarks...")
    from src.models.baselines import TextBlobSentiment
    from src.models.evaluator import ModelEvaluator

    evaluator = ModelEvaluator()
    texts = df["processed_text"].tolist()
    true_labels = df["sentiment"].tolist()

    evaluator.evaluate_model("VADER", vader, texts, true_labels)
    evaluator.evaluate_model("TextBlob", TextBlobSentiment(), texts, true_labels)

    comparison_df = evaluator.compare_models()
    agreement = evaluator.get_agreement_matrix(
        {"VADER": vader, "TextBlob": TextBlobSentiment()}, texts
    )

    # Save benchmarks so the dashboard can load them
    from pathlib import Path

    bench_dir = Path("data/benchmarks")
    bench_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(bench_dir / "comparison.csv", index=False)
    agreement.to_csv(bench_dir / "agreement.csv")
    print(f"  Benchmarks saved to {bench_dir}")

    print("Done! All dashboard pages should now have data.")
    print("Run: python -m src.dashboard.app")


if __name__ == "__main__":
    main()
