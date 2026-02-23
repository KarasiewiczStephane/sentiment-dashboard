"""Topics page with BERTopic visualization and sentiment breakdown."""

import logging

import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc

logger = logging.getLogger(__name__)

dash.register_page(__name__, path="/topics", name="Topics")

layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Topic Distribution"),
                            dbc.CardBody([dcc.Graph(id="topic-dist")]),
                        ]
                    ),
                    width=6,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Topic Sentiment Breakdown"),
                            dbc.CardBody([dcc.Graph(id="topic-sentiment")]),
                        ]
                    ),
                    width=6,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Topic Evolution"),
                            dbc.CardBody([dcc.Graph(id="topic-evolution")]),
                        ]
                    ),
                    width=12,
                )
            ]
        ),
    ]
)


@callback(
    [
        Output("topic-dist", "figure"),
        Output("topic-sentiment", "figure"),
        Output("topic-evolution", "figure"),
    ],
    Input("interval-component", "n_intervals"),
)
def update_topics(n_intervals: int) -> tuple:
    """Update topic visualization charts.

    Args:
        n_intervals: Auto-refresh interval counter.

    Returns:
        Tuple of (distribution figure, sentiment figure, evolution figure).
    """
    from src.dashboard.data_loader import load_sentiment_data, load_topic_model

    empty_fig = go.Figure()
    df = load_sentiment_data()
    topic_model = load_topic_model()

    if df.empty or topic_model is None:
        return empty_fig, empty_fig, empty_fig

    try:
        topic_df = topic_model.get_topic_distribution()
        dist_fig = px.bar(topic_df, x="Topic", y="Count", text="label")
    except Exception:
        dist_fig = empty_fig

    try:
        topic_sentiment = topic_model.get_topic_sentiment(
            df["processed_text"].tolist(), df["sentiment"].tolist()
        )
        sentiment_fig = px.bar(
            topic_sentiment.reset_index(),
            x="topic",
            y=["positive", "neutral", "negative"],
            barmode="stack",
        )
    except Exception:
        sentiment_fig = empty_fig

    try:
        if "created_at" in df.columns and df["created_at"].notna().any():
            evolution = topic_model.track_evolution(
                df["processed_text"].tolist(), df["created_at"].tolist()
            )
            evolution_fig = px.line(
                evolution.melt(
                    id_vars=["window_start"], var_name="topic", value_name="count"
                ),
                x="window_start",
                y="count",
                color="topic",
            )
        else:
            evolution_fig = empty_fig
    except Exception:
        evolution_fig = empty_fig

    return dist_fig, sentiment_fig, evolution_fig
