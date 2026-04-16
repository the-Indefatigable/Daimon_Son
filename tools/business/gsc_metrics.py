"""Google Search Console — clicks, impressions, CTR, position; top queries/pages.

Organic Google traffic visibility. Data lags ~2-3 days."""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

from permissions.levels import PermissionLevel
from tools.base import BaseTool


SITE_TO_ENV = {
    "centsibles": "GSC_SITE_CENTSIBLES",
    "fpl": "GSC_SITE_FPL",
    "quroots": "GSC_SITE_QUROOTS",
}


class GSCMetrics(BaseTool):
    name = "gsc_metrics"
    description = (
        "Pull Google Search Console data for Centsibles, FPL, or quroots. "
        "Returns organic clicks, impressions, average CTR and position, plus "
        "top queries and top landing pages. Data lags ~2-3 days. Use this "
        "to see: what are people ACTUALLY searching to find your site? "
        "Are we ranking for anything? Which queries convert to clicks?"
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
                    "description": "Lookback window. Default 28 (GSC has a "
                                   "2-3 day lag; shorter windows are noisy).",
                    "default": 28,
                },
                "dimension": {
                    "type": "string",
                    "enum": ["query", "page"],
                    "description": "Break top-results out by search query or "
                                   "landing page. Default 'query'.",
                    "default": "query",
                },
            },
            "required": ["site"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        site = str(kwargs.get("site", "")).strip()
        if site not in SITE_TO_ENV:
            return {"ok": False, "summary": f"unknown site: {site}"}
        site_url = os.getenv(SITE_TO_ENV[site], "").strip()
        if not site_url:
            return {"ok": False,
                    "summary": f"no {SITE_TO_ENV[site]} configured"}
        key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        if not key_path or not os.path.exists(key_path):
            return {"ok": False,
                    "summary": "GOOGLE_APPLICATION_CREDENTIALS missing or file not found"}

        days = int(kwargs.get("days", 28))
        dimension = kwargs.get("dimension", "query")
        end = date.today() - timedelta(days=3)  # account for GSC lag
        start = end - timedelta(days=days)

        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build

            creds = service_account.Credentials.from_service_account_file(
                key_path,
                scopes=["https://www.googleapis.com/auth/webmasters.readonly"],
            )
            svc = build("searchconsole", "v1", credentials=creds,
                        cache_discovery=False)

            # Totals (no dimensions)
            totals_resp = svc.searchanalytics().query(
                siteUrl=site_url,
                body={
                    "startDate": start.isoformat(),
                    "endDate": end.isoformat(),
                    "dimensions": [],
                    "rowLimit": 1,
                },
            ).execute()
            trow = (totals_resp.get("rows") or [{}])[0]
            clicks = int(trow.get("clicks", 0))
            impressions = int(trow.get("impressions", 0))
            ctr = round(float(trow.get("ctr", 0.0)) * 100, 2)
            pos = round(float(trow.get("position", 0.0)), 2)

            # Top results by chosen dimension
            top_resp = svc.searchanalytics().query(
                siteUrl=site_url,
                body={
                    "startDate": start.isoformat(),
                    "endDate": end.isoformat(),
                    "dimensions": [dimension],
                    "rowLimit": 15,
                    "orderBy": [{"field": "clicks", "descending": True}]
                    if False else None,
                },
            ).execute()
            rows = top_resp.get("rows", [])
            top = [
                {
                    dimension: r.get("keys", ["?"])[0],
                    "clicks": int(r.get("clicks", 0)),
                    "impressions": int(r.get("impressions", 0)),
                    "ctr_pct": round(float(r.get("ctr", 0)) * 100, 2),
                    "position": round(float(r.get("position", 0)), 2),
                }
                for r in rows
            ]

            return {
                "ok": True,
                "summary": (
                    f"{site} GSC last {days}d ({start}→{end}): "
                    f"{clicks} clicks, {impressions} impressions, "
                    f"CTR {ctr}%, avg pos {pos}."
                ),
                "site": site,
                "site_url": site_url,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "clicks": clicks,
                "impressions": impressions,
                "ctr_pct": ctr,
                "avg_position": pos,
                f"top_{dimension}s": top,
            }
        except Exception as e:
            return {"ok": False, "summary": f"gsc error: {type(e).__name__}: {e}"}
