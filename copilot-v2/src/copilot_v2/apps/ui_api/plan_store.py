from __future__ import annotations

from datetime import datetime, timezone


def save_plan_to_bigquery(*, plan_id: str, title: str) -> dict[str, str]:
    # Placeholder: this is where BigQuery insert logic would live.
    return {
        "status": "saved",
        "plan_id": plan_id,
        "title": title,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": "bigquery://placeholder_dataset.selected_plans",
    }

