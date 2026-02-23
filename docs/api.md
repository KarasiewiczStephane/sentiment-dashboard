# API Reference

The Sentiment Analysis API is built with FastAPI. Interactive documentation is available at `http://localhost:8000/docs` when the server is running.

## Endpoints

### POST /analyze

Analyze sentiment of a single text.

**Request:**
```json
{
  "text": "I love this product!"
}
```

**Response:**
```json
{
  "text": "I love this product!",
  "result": {
    "sentiment": "positive",
    "confidence": 0.95,
    "probabilities": {
      "positive": 0.95,
      "neutral": 0.03,
      "negative": 0.02
    }
  },
  "model": "roberta"
}
```

### POST /analyze/batch

Analyze sentiment of multiple texts (up to 100).

**Request:**
```json
{
  "texts": ["Great service!", "Terrible experience", "It was okay"]
}
```

**Response:**
```json
{
  "results": [
    {"text": "Great service!", "result": {"sentiment": "positive", "confidence": 0.92}, "model": "roberta"},
    {"text": "Terrible experience", "result": {"sentiment": "negative", "confidence": 0.88}, "model": "roberta"},
    {"text": "It was okay", "result": {"sentiment": "neutral", "confidence": 0.71}, "model": "roberta"}
  ],
  "total": 3,
  "processing_time_ms": 45.2
}
```

### GET /trends

Get sentiment trends for a specified period.

**Parameters:**
- `period` (string): Time period — `7d`, `24h`, `2w` etc.

**Response:**
```json
{
  "period": "7d",
  "data": [
    {"timestamp": "2024-01-01T00:00:00", "sentiment_score": 0.25, "volume": 150, "is_anomaly": false}
  ],
  "overall_trend": "stable",
  "avg_sentiment": 0.18
}
```

### GET /topics

List discovered topics with sentiment breakdown.

**Response:**
```json
{
  "topics": [
    {
      "topic_id": 0,
      "label": "customer, service, support",
      "count": 245,
      "dominant_sentiment": "positive",
      "positive_ratio": 0.65,
      "negative_ratio": 0.15
    }
  ],
  "total_topics": 8
}
```

### GET /health

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": {"roberta": true, "topic_model": true},
  "database_connected": true
}
```

### GET /export

Export filtered data as CSV.

**Parameters:**
- `start_date` (datetime, optional): Filter start date.
- `end_date` (datetime, optional): Filter end date.
- `sentiment` (string, optional): Filter by `positive`, `negative`, or `neutral`.
- `topic_id` (integer, optional): Filter by topic ID.

**Response:** CSV file download.
