"""Overview page with sentiment distribution and volume metrics."""

import logging

import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

logger = logging.getLogger(__name__)

dash.register_page(__name__, path="/", name="Overview")

layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Current Sentiment Score"),
                            dbc.CardBody(
                                [
                                    html.H2(
                                        id="sentiment-score",
                                        className="text-center",
                                    ),
                                    html.P(
                                        id="sentiment-label",
                                        className="text-center",
                                    ),
                                ]
                            ),
                        ]
                    ),
                    width=4,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Total Analyzed"),
                            dbc.CardBody(
                                [
                                    html.H2(
                                        id="total-count",
                                        className="text-center",
                                    )
                                ]
                            ),
                        ]
                    ),
                    width=4,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Positive Ratio"),
                            dbc.CardBody(
                                [
                                    html.H2(
                                        id="positive-ratio",
                                        className="text-center",
                                    )
                                ]
                            ),
                        ]
                    ),
                    width=4,
                ),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Sentiment Distribution"),
                            dbc.CardBody([dcc.Graph(id="sentiment-pie")]),
                        ]
                    ),
                    width=6,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Volume Over Time"),
                            dbc.CardBody([dcc.Graph(id="volume-line")]),
                        ]
                    ),
                    width=6,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [dcc.DatePickerRange(id="date-filter", className="mt-3")],
                    width=4,
                )
            ]
        ),
    ]
)


@callback(
    [
        Output("sentiment-score", "children"),
        Output("sentiment-label", "children"),
        Output("total-count", "children"),
        Output("positive-ratio", "children"),
        Output("sentiment-pie", "figure"),
        Output("volume-line", "figure"),
    ],
    [
        Input("interval-component", "n_intervals"),
        Input("date-filter", "start_date"),
        Input("date-filter", "end_date"),
    ],
)
def update_overview(n_intervals: int, start_date: str, end_date: str) -> tuple:
    """Update all overview page components.

    Args:
        n_intervals: Auto-refresh interval counter.
        start_date: Date filter start.
        end_date: Date filter end.

    Returns:
        Tuple of updated component values.
    """
    from src.dashboard.data_loader import load_sentiment_data

    df = load_sentiment_data(start_date, end_date)

    if df.empty:
        empty_fig = go.Figure()
        return "N/A", "No data", "0", "0%", empty_fig, empty_fig

    score = (df["sentiment"] == "positive").mean() - (
        df["sentiment"] == "negative"
    ).mean()
    label = "Positive" if score > 0.1 else ("Negative" if score < -0.1 else "Neutral")

    pie_fig = px.pie(
        df,
        names="sentiment",
        color="sentiment",
        color_discrete_map={
            "positive": "green",
            "neutral": "gray",
            "negative": "red",
        },
    )

    if "created_at" in df.columns and df["created_at"].notna().any():
        df["date"] = df["created_at"].dt.date
        volume = df.groupby("date").size().reset_index(name="count")
        line_fig = px.line(volume, x="date", y="count", title="Tweet Volume")
    else:
        line_fig = go.Figure()

    positive_ratio = (df["sentiment"] == "positive").mean() * 100
    return (
        f"{score:.2f}",
        label,
        f"{len(df):,}",
        f"{positive_ratio:.1f}%",
        pie_fig,
        line_fig,
    )
