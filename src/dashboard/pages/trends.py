"""Trends page with time series analysis and anomaly highlighting."""

import logging

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc

logger = logging.getLogger(__name__)

dash.register_page(__name__, path="/trends", name="Trends")

layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Sentiment Trends"),
                            dbc.CardBody(
                                [
                                    dcc.Dropdown(
                                        id="window-selector",
                                        options=[
                                            {"label": f"{w} days", "value": w}
                                            for w in [7, 14, 30, 90]
                                        ],
                                        value=7,
                                    ),
                                    dcc.Graph(id="trend-line"),
                                ]
                            ),
                        ]
                    ),
                    width=12,
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Anomaly Detection"),
                            dbc.CardBody([dcc.Graph(id="anomaly-chart")]),
                        ]
                    ),
                    width=12,
                )
            ]
        ),
    ]
)


@callback(
    [Output("trend-line", "figure"), Output("anomaly-chart", "figure")],
    [Input("interval-component", "n_intervals"), Input("window-selector", "value")],
)
def update_trends(n_intervals: int, window: int) -> tuple:
    """Update trend charts with current data and anomaly detection.

    Args:
        n_intervals: Auto-refresh interval counter.
        window: Rolling window size in days.

    Returns:
        Tuple of (trend figure, anomaly figure).
    """
    from src.analysis.trend_detector import TrendDetector
    from src.dashboard.data_loader import load_sentiment_data

    df = load_sentiment_data()

    if df.empty or "created_at" not in df.columns or df["created_at"].isna().all():
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[{"text": "No timestamp data available", "showarrow": False}]
        )
        return empty_fig, empty_fig

    detector = TrendDetector(window_size=window)
    anomalies = detector.detect_anomalies(
        df["sentiment"].tolist(), df["created_at"].tolist()
    )

    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=anomalies["period"],
            y=anomalies["sentiment_score"],
            mode="lines",
            name="Sentiment Score",
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=anomalies["period"],
            y=anomalies["rolling_mean"],
            mode="lines",
            name="Rolling Mean",
            line={"dash": "dash"},
        )
    )

    anomaly_df = anomalies[anomalies["is_anomaly"]]
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=anomalies["period"],
            y=anomalies["z_score"],
            mode="lines",
            name="Z-Score",
        )
    )
    fig2.add_hline(y=2, line_dash="dash", line_color="red")
    fig2.add_hline(y=-2, line_dash="dash", line_color="red")
    if not anomaly_df.empty:
        fig2.add_trace(
            go.Scatter(
                x=anomaly_df["period"],
                y=anomaly_df["z_score"],
                mode="markers",
                name="Anomalies",
                marker={"color": "red", "size": 10},
            )
        )

    return fig1, fig2
