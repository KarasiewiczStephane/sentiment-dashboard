"""RoBERTa-based sentiment classifier with fine-tuning pipeline."""

import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


class TweetDataset(Dataset):
    """PyTorch dataset for tokenized tweet data.

    Args:
        texts: List of tweet texts.
        labels: List of integer labels.
        tokenizer: RoBERTa tokenizer instance.
        max_length: Maximum sequence length for tokenization.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: RobertaTokenizer,
        max_length: int = 128,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


class RoBERTaSentimentClassifier:
    """Fine-tunable RoBERTa model for three-class sentiment classification.

    Args:
        model_name: Pretrained model identifier from HuggingFace.
        num_labels: Number of output classes.
        device: Torch device string ('cpu', 'cuda', or 'auto').
    """

    LABEL_MAP: dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
    LABEL_MAP_INV: dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_labels: int = 3,
        device: str = "auto",
    ) -> None:
        self.device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)
        self.num_labels = num_labels

    def train(
        self,
        train_texts: list[str],
        train_labels: list[str],
        val_texts: list[str],
        val_labels: list[str],
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        early_stopping_patience: int = 2,
        use_class_weights: bool = True,
        save_path: str = "models/roberta_best.pt",
    ) -> dict:
        """Fine-tune the model on labeled tweet data.

        Args:
            train_texts: Training text samples.
            train_labels: Training sentiment labels (string).
            val_texts: Validation text samples.
            val_labels: Validation sentiment labels (string).
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            learning_rate: Learning rate for AdamW optimizer.
            warmup_ratio: Fraction of total steps for warmup.
            early_stopping_patience: Epochs without improvement before stopping.
            use_class_weights: Whether to compute class weights for imbalanced data.
            save_path: File path to save the best model checkpoint.

        Returns:
            Training history dict with train_loss, val_loss, val_acc lists.
        """
        train_labels_int = [self.LABEL_MAP[label] for label in train_labels]
        val_labels_int = [self.LABEL_MAP[label] for label in val_labels]

        class_weights = None
        if use_class_weights:
            weights = compute_class_weight(
                "balanced", classes=np.unique(train_labels_int), y=train_labels_int
            )
            class_weights = torch.tensor(weights, dtype=torch.float).to(self.device)

        train_dataset = TweetDataset(train_texts, train_labels_int, self.tokenizer)
        val_dataset = TweetDataset(val_texts, val_labels_int, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * warmup_ratio),
            num_training_steps=total_steps,
        )

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_val_loss = float("inf")
        patience_counter = 0
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                )
                loss = criterion(outputs.logits, batch["labels"].to(self.device))
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()

            val_loss, val_acc = self._evaluate(val_loader, criterion)
            avg_train_loss = train_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            logger.info(
                "Epoch %d: train_loss=%.4f, val_loss=%.4f, val_acc=%.4f",
                epoch + 1,
                avg_train_loss,
                val_loss,
                val_acc,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save(save_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        return history

    def _evaluate(
        self, loader: DataLoader, criterion: nn.Module
    ) -> tuple[float, float]:
        """Evaluate the model on a data loader.

        Args:
            loader: DataLoader for evaluation data.
            criterion: Loss function.

        Returns:
            Tuple of (average loss, accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                )
                loss = criterion(outputs.logits, batch["labels"].to(self.device))
                total_loss += loss.item()
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == batch["labels"].to(self.device)).sum().item()
                total += len(batch["labels"])

        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
        accuracy = correct / total if total > 0 else 0
        return avg_loss, accuracy

    def predict(self, texts: list[str], return_confidence: bool = True) -> list[dict]:
        """Predict sentiment for a list of texts.

        Args:
            texts: List of input texts.
            return_confidence: Whether to include per-class probabilities.

        Returns:
            List of dicts with sentiment, confidence, and optional probabilities.
        """
        self.model.eval()
        dataset = TweetDataset(texts, [0] * len(texts), self.tokenizer)
        loader = DataLoader(dataset, batch_size=32)

        results = []
        with torch.no_grad():
            for batch in loader:
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                )
                probs = torch.softmax(outputs.logits, dim=-1)
                preds = probs.argmax(dim=-1)

                for i in range(len(preds)):
                    result: dict = {
                        "sentiment": self.LABEL_MAP_INV[preds[i].item()],
                        "confidence": probs[i].max().item(),
                    }
                    if return_confidence:
                        result["probabilities"] = {
                            self.LABEL_MAP_INV[j]: probs[i][j].item()
                            for j in range(self.num_labels)
                        }
                    results.append(result)
        return results

    def save(self, path: str) -> None:
        """Save model weights to disk.

        Args:
            path: File path for the saved model.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights from disk.

        Args:
            path: File path of the saved model.
        """
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        logger.info("Model loaded from %s", path)
