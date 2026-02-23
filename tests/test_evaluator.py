"""Tests for model evaluator and benchmarking."""

import pandas as pd
import pytest

from src.models.baselines import TextBlobSentiment, VADERSentiment
from src.models.evaluator import ModelEvaluator


class TestModelEvaluator:
    """Tests for the ModelEvaluator class."""

    def setup_method(self) -> None:
        """Set up evaluator and sample data."""
        self.evaluator = ModelEvaluator()
        self.texts = [
            "I absolutely love this product!",
            "This is terrible and horrible.",
            "The weather is okay today I guess.",
            "Amazing experience, highly recommend!",
            "Worst purchase I ever made.",
            "It works fine, nothing special.",
        ]
        self.true_labels = [
            "positive",
            "negative",
            "neutral",
            "positive",
            "negative",
            "neutral",
        ]

    def test_evaluate_vader(self) -> None:
        """VADER evaluation returns expected metric keys."""
        vader = VADERSentiment()
        metrics = self.evaluator.evaluate_model(
            "VADER", vader, self.texts, self.true_labels
        )
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "f1_weighted" in metrics
        assert "samples_per_second" in metrics
        assert "confusion_matrix" in metrics

    def test_evaluate_textblob(self) -> None:
        """TextBlob evaluation returns expected metric keys."""
        blob = TextBlobSentiment()
        metrics = self.evaluator.evaluate_model(
            "TextBlob", blob, self.texts, self.true_labels
        )
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_macro"] <= 1

    def test_accuracy_range(self) -> None:
        """Accuracy is between 0 and 1."""
        vader = VADERSentiment()
        metrics = self.evaluator.evaluate_model(
            "VADER", vader, self.texts, self.true_labels
        )
        assert 0 <= metrics["accuracy"] <= 1

    def test_inference_time_positive(self) -> None:
        """Inference time is a positive number."""
        vader = VADERSentiment()
        metrics = self.evaluator.evaluate_model(
            "VADER", vader, self.texts, self.true_labels
        )
        assert metrics["inference_time_total"] > 0
        assert metrics["samples_per_second"] > 0

    def test_confusion_matrix_shape(self) -> None:
        """Confusion matrix is 3x3 for 3-class classification."""
        vader = VADERSentiment()
        metrics = self.evaluator.evaluate_model(
            "VADER", vader, self.texts, self.true_labels
        )
        cm = metrics["confusion_matrix"]
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)

    def test_compare_models_returns_dataframe(self) -> None:
        """compare_models returns a DataFrame."""
        vader = VADERSentiment()
        blob = TextBlobSentiment()
        self.evaluator.evaluate_model("VADER", vader, self.texts, self.true_labels)
        self.evaluator.evaluate_model("TextBlob", blob, self.texts, self.true_labels)
        df = self.evaluator.compare_models()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "Model" in df.columns

    def test_agreement_matrix_symmetric(self) -> None:
        """Agreement matrix is symmetric with 1.0 on diagonal."""
        vader = VADERSentiment()
        blob = TextBlobSentiment()
        models = {"VADER": vader, "TextBlob": blob}
        matrix = self.evaluator.get_agreement_matrix(models, self.texts)
        assert isinstance(matrix, pd.DataFrame)
        assert matrix.loc["VADER", "VADER"] == pytest.approx(1.0)
        assert matrix.loc["TextBlob", "TextBlob"] == pytest.approx(1.0)
        assert matrix.loc["VADER", "TextBlob"] == pytest.approx(
            matrix.loc["TextBlob", "VADER"]
        )

    def test_results_stored(self) -> None:
        """Evaluated results are stored in the results dict."""
        vader = VADERSentiment()
        self.evaluator.evaluate_model("VADER", vader, self.texts, self.true_labels)
        assert "VADER" in self.evaluator.results
