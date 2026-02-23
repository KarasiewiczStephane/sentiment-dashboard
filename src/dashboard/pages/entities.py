"""Entities page with per-entity sentiment and trending entities."""

import logging

import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

logger = logging.getLogger(__name__)

dash.register_page(__name__, path="/entities", name="Entities")

layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Top Entities by Mentions"),
                            dbc.CardBody([dcc.Graph(id="entity-mentions")]),
                        ]
                    ),
                    width=6,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Trending Entities"),
                            dbc.CardBody([html.Div(id="trending-entities")]),
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
                            dbc.CardHeader("Entity Sentiment Cards"),
                            dbc.CardBody([html.Div(id="entity-cards")]),
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
        Output("entity-mentions", "figure"),
        Output("trending-entities", "children"),
        Output("entity-cards", "children"),
    ],
    Input("interval-component", "n_intervals"),
)
def update_entities(n_intervals: int) -> tuple:
    """Update entity analysis visualizations.

    Args:
        n_intervals: Auto-refresh interval counter.

    Returns:
        Tuple of (mentions figure, trending alerts, entity cards).
    """
    from src.dashboard.data_loader import load_sentiment_data

    empty_fig = go.Figure()
    df = load_sentiment_data()

    if df.empty or "processed_text" not in df.columns:
        return empty_fig, [], []

    try:
        from src.models.entity_analyzer import EntityAnalyzer

        analyzer = EntityAnalyzer()

        sentiments = [
            {"sentiment": s, "confidence": c if c else 0.5}
            for s, c in zip(
                df["sentiment"],
                df.get("confidence", [None] * len(df)),
            )
        ]

        entity_df = analyzer.aggregate_entity_sentiment(
            df["processed_text"].tolist(), sentiments, min_mentions=2
        )

        if entity_df.empty:
            return empty_fig, [html.P("No entities found")], []

        mentions_fig = px.bar(
            entity_df.head(20),
            x="entity",
            y="mentions",
            color="dominant_sentiment",
        )

        if "created_at" in df.columns and df["created_at"].notna().any():
            trends = analyzer.track_entity_trends(
                df["processed_text"].tolist(),
                sentiments,
                df["created_at"].tolist(),
            )
            trending = analyzer.get_trending_entities(trends)
        else:
            trending = []

        trending_cards = [
            dbc.Alert(
                f"{t['entity']}: {t['direction']} ({t['sentiment_change']:.2f})",
                color="success" if t["direction"] == "improving" else "danger",
            )
            for t in trending[:5]
        ]
        if not trending_cards:
            trending_cards = [html.P("No significant trends detected")]

        entity_cards = dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(row["entity"]),
                            dbc.CardBody(
                                [
                                    html.P(f"Mentions: {row['mentions']}"),
                                    html.P(f"Positive: {row['positive_ratio']:.1%}"),
                                    html.P(f"Negative: {row['negative_ratio']:.1%}"),
                                ]
                            ),
                        ]
                    ),
                    width=3,
                )
                for _, row in entity_df.head(8).iterrows()
            ]
        )

        return mentions_fig, trending_cards, entity_cards

    except Exception:
        logger.exception("Error updating entities page")
        return empty_fig, [html.P("Error loading entities")], []
