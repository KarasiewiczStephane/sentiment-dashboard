"""Model evaluation and benchmarking utilities."""

import logging
import time
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates and compares sentiment analysis models.

    Computes accuracy, F1 (macro/weighted), per-class metrics,
    confusion matrices, and inference speed.
    """

    LABELS: list[str] = ["negative", "neutral", "positive"]

    def __init__(self) -> None:
        self.results: dict[str, dict] = {}

    def evaluate_model(
        self,
        model_name: str,
        model: Any,
        texts: list[str],
        true_labels: list[str],
    ) -> dict:
        """Evaluate a single model against ground truth labels.

        Args:
            model_name: Human-readable name for the model.
            model: Model object with a predict(texts) method.
            texts: List of input texts.
            true_labels: List of ground truth sentiment labels.

        Returns:
            Dictionary of evaluation metrics.
        """
        start_time = time.time()
        predictions = model.predict(texts)
        inference_time = time.time() - start_time

        pred_labels = [p["sentiment"] for p in predictions]

        metrics = {
            "model": model_name,
            "accuracy": accuracy_score(true_labels, pred_labels),
            "f1_macro": f1_score(
                true_labels, pred_labels, average="macro", labels=self.LABELS
            ),
            "f1_weighted": f1_score(
                true_labels, pred_labels, average="weighted", labels=self.LABELS
            ),
            "precision_macro": precision_score(
                true_labels,
                pred_labels,
                average="macro",
                labels=self.LABELS,
                zero_division=0,
            ),
            "recall_macro": recall_score(
                true_labels,
                pred_labels,
                average="macro",
                labels=self.LABELS,
                zero_division=0,
            ),
            "inference_time_total": inference_time,
            "inference_time_per_sample": inference_time / len(texts),
            "samples_per_second": len(texts) / inference_time,
            "confusion_matrix": confusion_matrix(
                true_labels, pred_labels, labels=self.LABELS
            ).tolist(),
        }

        self.results[model_name] = metrics
        logger.info(
            "Evaluated %s: accuracy=%.3f, f1_macro=%.3f",
            model_name,
            metrics["accuracy"],
            metrics["f1_macro"],
        )
        return metrics

    def compare_models(self) -> pd.DataFrame:
        """Generate a benchmark comparison table.

        Returns:
            DataFrame with one row per model showing key metrics.
        """
        rows = []
        for name, metrics in self.results.items():
            rows.append(
                {
                    "Model": name,
                    "Accuracy": f"{metrics['accuracy']:.3f}",
                    "F1 (Macro)": f"{metrics['f1_macro']:.3f}",
                    "F1 (Weighted)": f"{metrics['f1_weighted']:.3f}",
                    "Samples/sec": f"{metrics['samples_per_second']:.1f}",
                }
            )
        return pd.DataFrame(rows)

    def get_agreement_matrix(
        self, models: dict[str, Any], texts: list[str]
    ) -> pd.DataFrame:
        """Calculate pairwise agreement between models.

        Args:
            models: Dict mapping model names to model instances.
            texts: List of input texts.

        Returns:
            DataFrame with agreement ratios between all model pairs.
        """
        predictions = {
            name: [p["sentiment"] for p in model.predict(texts)]
            for name, model in models.items()
        }
        names = list(models.keys())
        matrix = []
        for n1 in names:
            row = []
            for n2 in names:
                agreement = sum(
                    p1 == p2 for p1, p2 in zip(predictions[n1], predictions[n2])
                ) / len(texts)
                row.append(agreement)
            matrix.append(row)
        return pd.DataFrame(matrix, index=names, columns=names)
