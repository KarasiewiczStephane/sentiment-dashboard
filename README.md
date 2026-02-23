# Sentiment Analysis Dashboard

Real-time sentiment monitoring with fine-tuned RoBERTa, BERTopic topic modeling, and interactive Plotly Dash visualization.

[![CI](https://github.com/KarasiewiczStephane/sentiment-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/KarasiewiczStephane/sentiment-dashboard/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Overview

A comprehensive sentiment analysis system featuring:

- **Fine-tuned RoBERTa** for accurate sentiment classification (positive/negative/neutral)
- **VADER and TextBlob** baselines with benchmark comparison
- **BERTopic** for automatic topic discovery and evolution tracking
- **Entity-level sentiment** analysis using spaCy NER
- **Z-score anomaly detection** and statistical trend analysis
- **Real-time dashboard** with Plotly Dash (5 pages)
- **REST API** with FastAPI for integration
- **Executive reports** with HTML/PDF export
- **Docker** containerization with CI/CD pipeline

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Data Source  │────>│  Processing  │────>│   Storage    │
│  (TweetEval) │     │   Pipeline   │     │   (DuckDB)   │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                 ┌────────────────┬────────────────┤
                 │                │                │
                 v                v                v
          ┌──────────┐    ┌──────────┐    ┌──────────┐
          │ RoBERTa  │    │ BERTopic │    │ spaCy    │
          │Classifier│    │ Topics   │    │ NER      │
          └────┬─────┘    └────┬─────┘    └────┬─────┘
               │               │               │
               └───────────────┼───────────────┘
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
                 v             v             v
          ┌──────────┐  ┌──────────┐  ┌──────────┐
          │   Dash   │  │ FastAPI  │  │  Report  │
          │Dashboard │  │   API    │  │Generator │
          └──────────┘  └──────────┘  └──────────┘
```

## Quick Start

```bash
git clone https://github.com/KarasiewiczStephane/sentiment-dashboard.git
cd sentiment-dashboard

# Install dependencies
make install

# Run tests
make test

# Start API server
make run-api        # http://localhost:8000

# Start dashboard
make run-dashboard  # http://localhost:8050
```

### Docker

```bash
docker-compose up --build
```

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Overview** | Sentiment score, distribution pie chart, volume over time |
| **Trends** | Time series with configurable windows, anomaly highlighting |
| **Topics** | BERTopic topic distribution, sentiment breakdown, evolution |
| **Entities** | Per-entity sentiment cards, trending entity alerts |
| **Model Comparison** | Benchmark table, agreement heatmap, speed comparison |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Single text sentiment analysis |
| `/analyze/batch` | POST | Batch analysis (up to 100 texts) |
| `/trends` | GET | Sentiment trends with anomaly detection |
| `/topics` | GET | Topic listing with sentiment breakdown |
| `/export` | GET | Export filtered data as CSV |
| `/health` | GET | Health check |

Full API docs at `http://localhost:8000/docs` (Swagger UI) or [docs/api.md](docs/api.md).

### Example

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

## Project Structure

```
sentiment-dashboard/
├── src/
│   ├── analysis/      # Trend detection, report generation
│   ├── api/           # FastAPI REST endpoints
│   ├── dashboard/     # Plotly Dash app and pages
│   ├── data/          # Data loading, preprocessing, simulation
│   ├── models/        # RoBERTa, baselines, topics, entities
│   └── utils/         # Config, logging
├── tests/             # Unit tests (190+ tests, 87%+ coverage)
├── configs/           # YAML configuration
├── templates/         # Jinja2 report templates
├── data/sample/       # Sample dataset for testing
├── docs/              # API documentation
├── .github/workflows/ # CI pipeline
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

## Testing

```bash
make test       # Run tests with coverage
make lint       # Check linting
make lint-fix   # Auto-fix lint issues
```

## License

MIT
