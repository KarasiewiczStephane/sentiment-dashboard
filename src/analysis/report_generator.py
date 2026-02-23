"""Executive report generation with Jinja2 templates and optional PDF export."""

import base64
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates sentiment analysis reports in HTML and PDF formats.

    Args:
        template_dir: Directory containing Jinja2 templates.
    """

    def __init__(self, template_dir: str = "templates") -> None:
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.template = self.env.get_template("report_template.html")

    def _fig_to_base64(self, fig: object) -> str:
        """Convert a Plotly figure to a base64-encoded PNG string.

        Args:
            fig: Plotly figure object.

        Returns:
            Base64-encoded PNG string.
        """
        try:
            import plotly.io as pio

            img_bytes = pio.to_image(fig, format="png", scale=2)
            return base64.b64encode(img_bytes).decode("utf-8")
        except Exception:
            logger.warning("Could not convert figure to image")
            return ""

    def generate_report(
        self,
        sentiment_data: dict,
        topic_data: Optional[dict] = None,
        entity_data: Optional[dict] = None,
        trend_data: Optional[dict] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> str:
        """Generate an HTML report from provided data sections.

        Args:
            sentiment_data: Dict with total, avg_score, counts, and ratios.
            topic_data: Optional dict with positive_topics and negative_topics lists.
            entity_data: Optional dict with trending entity list.
            trend_data: Optional dict with dates, scores, summary, and anomalies.
            start_date: Report period start.
            end_date: Report period end.

        Returns:
            Rendered HTML string.
        """
        now = datetime.now()

        context = {
            "report_date": now.strftime("%Y-%m-%d %H:%M"),
            "start_date": start_date.strftime("%Y-%m-%d") if start_date else "N/A",
            "end_date": end_date.strftime("%Y-%m-%d") if end_date else "N/A",
            "total_analyzed": sentiment_data.get("total", 0),
            "avg_sentiment": sentiment_data.get("avg_score", 0),
            "positive_ratio": sentiment_data.get("positive_ratio", 0),
            "negative_ratio": sentiment_data.get("negative_ratio", 0),
            "sentiment_pie_chart": None,
            "positive_topics": [],
            "negative_topics": [],
            "trending_entities": [],
            "trend_chart": None,
            "trend_summary": "",
            "anomalies": [],
        }

        if topic_data:
            context["positive_topics"] = topic_data.get("positive_topics", [])[:5]
            context["negative_topics"] = topic_data.get("negative_topics", [])[:5]

        if entity_data:
            context["trending_entities"] = entity_data.get("trending", [])[:10]

        if trend_data:
            context["trend_summary"] = trend_data.get("summary", "")
            context["anomalies"] = trend_data.get("anomalies", [])

        return self.template.render(**context)

    def save_html(self, html_content: str, output_path: str) -> str:
        """Save rendered HTML to a file.

        Args:
            html_content: Rendered HTML string.
            output_path: Output file path.

        Returns:
            The output file path.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html_content)
        logger.info("HTML report saved to %s", output_path)
        return output_path

    def save_pdf(self, html_content: str, output_path: str) -> str:
        """Convert HTML to PDF using WeasyPrint.

        Args:
            html_content: Rendered HTML string.
            output_path: Output PDF file path.

        Returns:
            The output file path.
        """
        try:
            from weasyprint import HTML

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            HTML(string=html_content).write_pdf(output_path)
            logger.info("PDF report saved to %s", output_path)
        except ImportError:
            logger.warning("WeasyPrint not available, saving as HTML instead")
            output_path = output_path.replace(".pdf", ".html")
            self.save_html(html_content, output_path)
        return output_path
