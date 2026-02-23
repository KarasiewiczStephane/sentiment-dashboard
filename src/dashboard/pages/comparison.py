"""Model comparison page with benchmark results and agreement heatmap."""

import logging

import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

logger = logging.getLogger(__name__)

dash.register_page(__name__, path="/comparison", name="Model Comparison")

layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Model Performance Comparison"),
                            dbc.CardBody([html.Div(id="benchmark-table")]),
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
                            dbc.CardHeader("Model Agreement Heatmap"),
                            dbc.CardBody([dcc.Graph(id="agreement-heatmap")]),
                        ]
                    ),
                    width=6,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Inference Speed"),
                            dbc.CardBody([dcc.Graph(id="speed-chart")]),
                        ]
                    ),
                    width=6,
                ),
            ]
        ),
    ]
)


@callback(
    [
        Output("benchmark-table", "children"),
        Output("agreement-heatmap", "figure"),
        Output("speed-chart", "figure"),
    ],
    Input("interval-component", "n_intervals"),
)
def update_comparison(n_intervals: int) -> tuple:
    """Update model comparison visualizations.

    Args:
        n_intervals: Auto-refresh interval counter.

    Returns:
        Tuple of (benchmark table, agreement heatmap, speed chart).
    """
    from src.dashboard.data_loader import load_benchmark_results

    results = load_benchmark_results()
    empty_fig = go.Figure()

    comparison_df = results.get("comparison_df")
    agreement_matrix = results.get("agreement_matrix")

    if comparison_df is None or comparison_df.empty:
        return html.P("No benchmark results available"), empty_fig, empty_fig

    table = dbc.Table.from_dataframe(comparison_df, striped=True, bordered=True)

    if agreement_matrix is not None and not agreement_matrix.empty:
        agreement_fig = go.Figure(
            data=go.Heatmap(
                z=agreement_matrix.values,
                x=agreement_matrix.columns.tolist(),
                y=agreement_matrix.index.tolist(),
                colorscale="Viridis",
                text=agreement_matrix.round(3).values,
                texttemplate="%{text}",
            )
        )
    else:
        agreement_fig = empty_fig

    if "Samples/sec" in comparison_df.columns:
        speed_fig = px.bar(
            comparison_df,
            x="Model",
            y="Samples/sec",
            title="Inference Speed (samples/second)",
        )
    else:
        speed_fig = empty_fig

    return table, agreement_fig, speed_fig
