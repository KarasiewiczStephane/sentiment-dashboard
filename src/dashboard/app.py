"""Plotly Dash multi-page application for sentiment analysis."""

import logging

import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc

logger = logging.getLogger(__name__)


def create_app() -> Dash:
    """Create and configure the Dash application.

    Returns:
        Configured Dash app instance.
    """
    app = Dash(
        __name__,
        use_pages=True,
        pages_folder="pages",
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )

    app.layout = dbc.Container(
        [
            dbc.NavbarSimple(
                children=[
                    dbc.NavItem(dbc.NavLink("Overview", href="/")),
                    dbc.NavItem(dbc.NavLink("Trends", href="/trends")),
                    dbc.NavItem(dbc.NavLink("Topics", href="/topics")),
                    dbc.NavItem(dbc.NavLink("Entities", href="/entities")),
                    dbc.NavItem(dbc.NavLink("Model Comparison", href="/comparison")),
                ],
                brand="Sentiment Dashboard",
                brand_href="/",
                color="primary",
                dark=True,
            ),
            dcc.Interval(id="interval-component", interval=300 * 1000, n_intervals=0),
            dcc.Store(id="data-store"),
            dash.page_container,
        ],
        fluid=True,
    )

    return app


app = create_app()
server = app.server

if __name__ == "__main__":
    app.run(debug=True, port=8050)
