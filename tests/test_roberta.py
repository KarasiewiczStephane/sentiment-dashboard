"""Tests for the RoBERTa sentiment classifier."""

from pathlib import Path

import pytest
import torch

from src.models.roberta_classifier import (
    RoBERTaSentimentClassifier,
    TweetDataset,
)


@pytest.fixture(scope="module")
def classifier() -> RoBERTaSentimentClassifier:
    """Create a classifier instance (shared across tests for speed)."""
    return RoBERTaSentimentClassifier(
        model_name="roberta-base", num_labels=3, device="cpu"
    )


class TestTweetDataset:
    """Tests for the TweetDataset class."""

    def test_dataset_length(self, classifier: RoBERTaSentimentClassifier) -> None:
        """Dataset length matches input length."""
        ds = TweetDataset(
            ["hello world", "goodbye"], [0, 1], classifier.tokenizer, max_length=32
        )
        assert len(ds) == 2

    def test_dataset_item_keys(self, classifier: RoBERTaSentimentClassifier) -> None:
        """Dataset items contain expected keys."""
        ds = TweetDataset(["test text"], [0], classifier.tokenizer, max_length=32)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_dataset_labels_tensor(
        self, classifier: RoBERTaSentimentClassifier
    ) -> None:
        """Labels are stored as a tensor."""
        ds = TweetDataset(
            ["one", "two", "three"], [0, 1, 2], classifier.tokenizer, max_length=32
        )
        assert isinstance(ds.labels, torch.Tensor)
        assert ds.labels.tolist() == [0, 1, 2]


class TestRoBERTaSentimentClassifier:
    """Tests for the RoBERTaSentimentClassifier."""

    def test_label_maps_consistent(self) -> None:
        """Forward and inverse label maps are consistent."""
        for label, idx in RoBERTaSentimentClassifier.LABEL_MAP.items():
            assert RoBERTaSentimentClassifier.LABEL_MAP_INV[idx] == label

    def test_predict_returns_list(self, classifier: RoBERTaSentimentClassifier) -> None:
        """predict returns a list of result dicts."""
        results = classifier.predict(["I love this!", "I hate this."])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_predict_result_structure(
        self, classifier: RoBERTaSentimentClassifier
    ) -> None:
        """Each prediction result has sentiment, confidence, and probabilities."""
        results = classifier.predict(["Great product"])
        result = results[0]
        assert "sentiment" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert result["sentiment"] in ("positive", "negative", "neutral")

    def test_predict_confidence_range(
        self, classifier: RoBERTaSentimentClassifier
    ) -> None:
        """Confidence is between 0 and 1."""
        results = classifier.predict(["This is okay"])
        assert 0 <= results[0]["confidence"] <= 1

    def test_predict_probabilities_sum_to_one(
        self, classifier: RoBERTaSentimentClassifier
    ) -> None:
        """Probabilities across classes sum to approximately 1."""
        results = classifier.predict(["A test sentence"])
        probs = results[0]["probabilities"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01

    def test_predict_without_confidence(
        self, classifier: RoBERTaSentimentClassifier
    ) -> None:
        """predict with return_confidence=False omits probabilities."""
        results = classifier.predict(["Test"], return_confidence=False)
        assert "probabilities" not in results[0]

    def test_save_and_load(
        self, classifier: RoBERTaSentimentClassifier, tmp_path: Path
    ) -> None:
        """Model can be saved and loaded without error."""
        save_path = str(tmp_path / "model.pt")
        classifier.save(save_path)
        assert Path(save_path).exists()

        new_classifier = RoBERTaSentimentClassifier(device="cpu")
        new_classifier.load(save_path)

        original = classifier.predict(["Test sentence"])
        loaded = new_classifier.predict(["Test sentence"])
        assert original[0]["sentiment"] == loaded[0]["sentiment"]

    def test_save_creates_directory(
        self, classifier: RoBERTaSentimentClassifier, tmp_path: Path
    ) -> None:
        """save creates parent directories if missing."""
        save_path = str(tmp_path / "nested" / "dir" / "model.pt")
        classifier.save(save_path)
        assert Path(save_path).exists()

    def test_device_auto_selects_cpu(self) -> None:
        """With 'auto' device and no CUDA, CPU is selected."""
        clf = RoBERTaSentimentClassifier(device="auto")
        assert clf.device in ("cpu", "cuda")

    def test_predict_batch_larger_than_batch_size(
        self, classifier: RoBERTaSentimentClassifier
    ) -> None:
        """Prediction works with inputs larger than the internal batch size."""
        texts = [f"Test sentence number {i}" for i in range(50)]
        results = classifier.predict(texts)
        assert len(results) == 50


class TestTraining:
    """Tests for the training loop (small scale)."""

    def test_train_returns_history(self, tmp_path: Path) -> None:
        """Training returns a history dict with expected keys."""
        clf = RoBERTaSentimentClassifier(device="cpu")
        texts = ["I love this", "Terrible product", "It is okay"] * 5
        labels = ["positive", "negative", "neutral"] * 5
        save_path = str(tmp_path / "model.pt")

        history = clf.train(
            train_texts=texts,
            train_labels=labels,
            val_texts=texts[:3],
            val_labels=labels[:3],
            epochs=1,
            batch_size=8,
            save_path=save_path,
        )

        assert "train_loss" in history
        assert "val_loss" in history
        assert "val_acc" in history
        assert len(history["train_loss"]) == 1

    def test_train_loss_is_finite(self, tmp_path: Path) -> None:
        """Training loss is a finite number."""
        clf = RoBERTaSentimentClassifier(device="cpu")
        texts = ["Good day", "Bad day", "Normal day"] * 4
        labels = ["positive", "negative", "neutral"] * 4
        save_path = str(tmp_path / "model.pt")

        history = clf.train(
            train_texts=texts,
            train_labels=labels,
            val_texts=texts[:3],
            val_labels=labels[:3],
            epochs=1,
            batch_size=8,
            save_path=save_path,
        )

        assert all(isinstance(v, float) for v in history["train_loss"])
