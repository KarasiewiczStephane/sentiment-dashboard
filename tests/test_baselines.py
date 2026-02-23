"""Tests for baseline sentiment models."""

from src.models.baselines import TextBlobSentiment, VADERSentiment


class TestVADERSentiment:
    """Tests for the VADER baseline model."""

    def setup_method(self) -> None:
        """Create VADER instance for each test."""
        self.vader = VADERSentiment()

    def test_predict_returns_list(self) -> None:
        """predict returns a list of dicts."""
        results = self.vader.predict(["I love this", "I hate this"])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_positive_sentiment(self) -> None:
        """Clearly positive text is classified as positive."""
        results = self.vader.predict(["This is absolutely wonderful and amazing!"])
        assert results[0]["sentiment"] == "positive"

    def test_negative_sentiment(self) -> None:
        """Clearly negative text is classified as negative."""
        results = self.vader.predict(["This is terrible and awful, worst ever!"])
        assert results[0]["sentiment"] == "negative"

    def test_result_has_scores(self) -> None:
        """Result includes raw VADER scores."""
        results = self.vader.predict(["Hello world"])
        assert "scores" in results[0]
        assert "compound" in results[0]["scores"]

    def test_confidence_range(self) -> None:
        """Confidence is between 0 and 1."""
        results = self.vader.predict(["Great stuff"])
        assert 0 <= results[0]["confidence"] <= 1

    def test_compound_to_label_thresholds(self) -> None:
        """Label thresholds map correctly."""
        assert self.vader._compound_to_label(0.5) == "positive"
        assert self.vader._compound_to_label(-0.5) == "negative"
        assert self.vader._compound_to_label(0.0) == "neutral"
        assert self.vader._compound_to_label(0.04) == "neutral"
        assert self.vader._compound_to_label(-0.04) == "neutral"


class TestTextBlobSentiment:
    """Tests for the TextBlob baseline model."""

    def setup_method(self) -> None:
        """Create TextBlob instance for each test."""
        self.blob = TextBlobSentiment()

    def test_predict_returns_list(self) -> None:
        """predict returns a list of dicts."""
        results = self.blob.predict(["I love this", "I hate this"])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_positive_sentiment(self) -> None:
        """Clearly positive text is classified as positive."""
        results = self.blob.predict(["This is absolutely wonderful and amazing!"])
        assert results[0]["sentiment"] == "positive"

    def test_negative_sentiment(self) -> None:
        """Clearly negative text is classified as negative."""
        results = self.blob.predict(["This is terrible and awful!"])
        assert results[0]["sentiment"] == "negative"

    def test_result_has_polarity(self) -> None:
        """Result includes polarity score."""
        results = self.blob.predict(["Hello world"])
        assert "polarity" in results[0]

    def test_result_has_subjectivity(self) -> None:
        """Result includes subjectivity score."""
        results = self.blob.predict(["This is amazing"])
        assert "subjectivity" in results[0]

    def test_polarity_to_label_thresholds(self) -> None:
        """Label thresholds map correctly."""
        assert self.blob._polarity_to_label(0.5) == "positive"
        assert self.blob._polarity_to_label(-0.5) == "negative"
        assert self.blob._polarity_to_label(0.0) == "neutral"
