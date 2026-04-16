"""GA4 Data API — sessions, users, pageviews, top pages/sources for each site.

Uses the service account at GOOGLE_APPLICATION_CREDENTIALS."""
from __future__ import annotations

import os
from typing import Any

from permissions.levels import PermissionLevel
from tools.base import BaseTool


SITE_TO_ENV = {
    "centsibles": "GA4_PROPERTY_CENTSIBLES",
    "fpl": "GA4_PROPERTY_FPL",
    "quroots": "GA4_PROPERTY_QUROOTS",
}


class GA4Metrics(BaseTool):
    name = "ga4_metrics"
    description = (
        "Pull Google Analytics 4 data for Centsibles, FPL, or quroots. "
        "Returns sessions, users, pageviews, bounce rate, top pages, top "
        "traffic sources over a lookback window. Use this to see whether "
        "your SEO/content/PR work actually moved traffic. Compare pre/post."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "site": {
                    "type": "string",
                    "enum": ["centsibles", "fpl", "quroots"],
                },
                "days": {
                    "type": "integer",
                    "description": "Lookback window. Default 7.",
                    "default": 7,
                },
            },
            "required": ["site"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        site = str(kwargs.get("site", "")).strip()
        if site not in SITE_TO_ENV:
            return {"ok": False, "summary": f"unknown site: {site}"}
        property_id = os.getenv(SITE_TO_ENV[site], "").strip()
        if not property_id:
            return {"ok": False,
                    "summary": f"no {SITE_TO_ENV[site]} configured"}
        key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        if not key_path or not os.path.exists(key_path):
            return {"ok": False,
                    "summary": "GOOGLE_APPLICATION_CREDENTIALS missing or file not found"}

        days = int(kwargs.get("days", 7))
        try:
            from google.analytics.data_v1beta import BetaAnalyticsDataClient
            from google.analytics.data_v1beta.types import (
                DateRange, Dimension, Metric, RunReportRequest,
            )
            from google.oauth2 import service_account

            creds = service_account.Credentials.from_service_account_file(
                key_path,
                scopes=["https://www.googleapis.com/auth/analytics.readonly"],
            )
            client = BetaAnalyticsDataClient(credentials=creds)
            date_range = DateRange(start_date=f"{days}daysAgo", end_date="today")

            # Totals
            totals_req = RunReportRequest(
                property=f"properties/{property_id}",
                date_ranges=[date_range],
                metrics=[
                    Metric(name="sessions"),
                    Metric(name="totalUsers"),
                    Metric(name="screenPageViews"),
                    Metric(name="bounceRate"),
                    Metric(name="averageSessionDuration"),
                ],
            )
            totals = client.run_report(totals_req)
            tvals = [v.value for v in totals.rows[0].metric_values] if totals.rows else ["0"] * 5

            # Top pages
            pages_req = RunReportRequest(
                property=f"properties/{property_id}",
                date_ranges=[date_range],
                dimensions=[Dimension(name="pagePath")],
                metrics=[Metric(name="screenPageViews")],
                limit=10,
            )
            pages = client.run_report(pages_req)
            top_pages = [
                {"path": r.dimension_values[0].value,
                 "views": int(r.metric_values[0].value)}
                for r in pages.rows
            ]

            # Top sources
            src_req = RunReportRequest(
                property=f"properties/{property_id}",
                date_ranges=[date_range],
                dimensions=[Dimension(name="sessionSource"),
                            Dimension(name="sessionMedium")],
                metrics=[Metric(name="sessions")],
                limit=10,
            )
            src = client.run_report(src_req)
            top_sources = [
                {"source": r.dimension_values[0].value,
                 "medium": r.dimension_values[1].value,
                 "sessions": int(r.metric_values[0].value)}
                for r in src.rows
            ]

            sessions = int(tvals[0])
            users = int(tvals[1])
            pageviews = int(tvals[2])
            bounce = round(float(tvals[3]) * 100, 1)
            avg_dur = round(float(tvals[4]), 1)

            return {
                "ok": True,
                "summary": (
                    f"{site} last {days}d: {sessions} sessions, "
                    f"{users} users, {pageviews} pageviews, "
                    f"bounce {bounce}%, avg dur {avg_dur}s."
                ),
                "site": site,
                "window_days": days,
                "sessions": sessions,
                "users": users,
                "pageviews": pageviews,
                "bounce_rate_pct": bounce,
                "avg_session_duration_s": avg_dur,
                "top_pages": top_pages,
                "top_sources": top_sources,
            }
        except Exception as e:
            return {"ok": False, "summary": f"ga4 error: {type(e).__name__}: {e}"}
