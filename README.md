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

# 1. Install dependencies (includes spaCy en_core_web_sm model)
make install

# 2. Download TweetEval data, preprocess, and populate the DuckDB database
#    This downloads from Hugging Face, runs VADER scoring, spaCy NER,
#    BERTopic fitting, and saves benchmark results (~2-5 min)
python scripts/populate_demo.py

# 3. Launch the dashboard
make run-dashboard  # http://localhost:8050
```

To start the API server separately:

```bash
make run-api        # http://localhost:8000/docs (Swagger UI)
```

> **Note:** The dashboard reads from `data/sentiment.duckdb`. You must run
> `populate_demo.py` (step 2) at least once before launching the dashboard
> or API, otherwise pages will show empty data.

### Docker

```bash
docker-compose up --build
# API  → http://localhost:8000
# Dash → http://localhost:8050
```

The `docker-compose.yml` starts two services: **api** (FastAPI on port 8000) and **dashboard** (Plotly Dash on port 8050). Both mount `data/` and `models/` as volumes for persistence.

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Overview** (`/`) | Sentiment score, distribution pie chart, volume over time |
| **Trends** (`/trends`) | Time series with configurable windows, anomaly highlighting |
| **Topics** (`/topics`) | BERTopic topic distribution, sentiment breakdown, evolution |
| **Entities** (`/entities`) | Per-entity sentiment cards, trending entity alerts |
| **Model Comparison** (`/comparison`) | Benchmark table, agreement heatmap, speed comparison |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Single text sentiment analysis |
| `/analyze/batch` | POST | Batch analysis (up to 100 texts) |
| `/trends` | GET | Sentiment trends with anomaly detection |
| `/topics` | GET | Topic listing with sentiment breakdown |
| `/export` | GET | Export filtered data as CSV |
| `/health` | GET | Health check |

Full interactive docs available at `http://localhost:8000/docs` (Swagger UI).

### Example

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

## Data Pipeline

The data pipeline works in stages:

1. **Download** -- `src/data/downloader.py` fetches the TweetEval sentiment dataset from Hugging Face (`tweet_eval/sentiment`).
2. **Preprocess** -- `src/data/preprocessor.py` cleans text (URL/mention removal, emoji conversion, language filtering) and stores results in DuckDB via `src/data/database.py`.
3. **Enrich** -- `scripts/populate_demo.py` adds timestamps, VADER scores, spaCy entities, and BERTopic topic assignments to the database. Benchmark results are saved to `data/benchmarks/`.
4. **Simulate** -- `src/data/simulator.py` can replay stored tweets as a real-time feed for dashboard testing.

You can also generate a standalone sample CSV (no DB required):

```bash
python data/sample/generate_sample.py  # Writes data/sample/sample_tweets.csv
```

## Project Structure

```
sentiment-dashboard/
├── src/
│   ├── analysis/          # Trend detection, report generation
│   ├── api/               # FastAPI REST endpoints and Pydantic schemas
│   ├── dashboard/         # Plotly Dash app with multi-page layout
│   │   └── pages/         # Overview, Trends, Topics, Entities, Comparison
│   ├── data/              # Downloader, preprocessor, DuckDB, simulator
│   ├── models/            # RoBERTa, VADER/TextBlob baselines, BERTopic, NER
│   └── utils/             # Config loader, logger
├── tests/                 # 194 unit tests across 18 modules
├── scripts/               # populate_demo.py (full data pipeline)
├── data/
│   └── sample/            # generate_sample.py for standalone CSV
├── .github/workflows/     # CI pipeline (lint, test, docker)
├── Dockerfile             # Multi-stage Python 3.11-slim build
├── docker-compose.yml     # API + Dashboard services
├── Makefile
├── requirements.txt
└── pyproject.toml
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make install` | Install Python dependencies and spaCy model |
| `make test` | Run pytest with coverage report |
| `make lint` | Check code with ruff |
| `make lint-fix` | Auto-fix lint issues |
| `make run-api` | Start FastAPI server (port 8000) |
| `make run-dashboard` | Start Plotly Dash app (port 8050) |
| `make docker-compose-up` | Build and start all services |
| `make clean` | Remove caches and coverage files |

## Testing

```bash
make test       # Run all tests with coverage
make lint       # Check linting with ruff
make lint-fix   # Auto-fix lint issues
```

## License

MIT
