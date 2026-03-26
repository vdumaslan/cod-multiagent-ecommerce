from __future__ import annotations

from typing import Any

import pandas as pd


def apply_policy(
    plans: list[dict[str, Any]],
    inventory: pd.DataFrame,
    *,
    min_margin: float = 0.08,
    max_abs_price_change: float = 10.0,
) -> tuple[list[dict[str, Any]], list[str]]:
    margin_map = dict(zip(inventory["product_id"], inventory["margin_pct"]))
    kept: list[dict[str, Any]] = []
    warnings: list[str] = []

    for p in plans:
        if not p.get("evidence_refs"):
            warnings.append(f"{p['plan_id']}: missing evidence refs")
            continue
        if abs(float(p.get("price_change_pct", 0.0))) > max_abs_price_change:
            warnings.append(f"{p['plan_id']}: price change cap exceeded")
            continue
        skus = p.get("impacted_skus", [])
        margins = [margin_map.get(s) for s in skus if margin_map.get(s) is not None]
        if margins and float(pd.Series(margins).mean()) < min_margin:
            warnings.append(f"{p['plan_id']}: below margin floor")
            continue
        kept.append(p)

    if not kept and plans:
        kept = [plans[0]]
        warnings.append("No plan fully passed policy; returning safest fallback plan.")
    return kept, warnings
