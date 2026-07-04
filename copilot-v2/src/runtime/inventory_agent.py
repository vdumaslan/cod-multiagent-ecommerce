from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class InventoryAgentConfig:
    # Thresholds are intentionally simple and deterministic.
    low_stock_available_units: float = 5.0
    overstock_on_hand_units: float = 50.0
    # When revenue is very low, inventory is more likely dead/overstock.
    low_revenue_usd_per_day: float = 1.0
    # Returns are treated as an extra risk signal.
    high_returns_units: float = 10.0


def _to_num(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def classify_inventory_for_row(
    row: dict[str, Any],
    *,
    cfg: InventoryAgentConfig,
) -> dict[str, Any]:
    """
    Classify inventory status for one candidate row.

    Expected inputs (best effort; missing values default to 0):
    - From either top-level keys OR nested `signals` dict:
      - on_hand_units, safety_stock_units, available_to_sell (optional; derived if missing)
      - mean_daily_revenue, total_returns
    """
    signals = row.get("signals") if isinstance(row.get("signals"), dict) else {}

    def _get(key: str) -> Any:
        if key in row:
            return row.get(key)
        return (signals or {}).get(key)

    on_hand = _to_num(_get("on_hand_units"))
    safety = _to_num(_get("safety_stock_units"))
    available = _get("available_to_sell")
    if available is None:
        available = max(on_hand - safety, 0.0)
    else:
        available = _to_num(available)

    mean_daily_revenue = _to_num(_get("mean_daily_revenue"))
    total_returns = _to_num(_get("total_returns"))

    rules: list[str] = []
    risk_flag = False

    # 4-class contract: stockout_risk | low_stock | overstocked | healthy
    if available <= 0.0 or on_hand <= safety:
        stock_status = "stockout_risk"
        risk_flag = True
        rules.append("stockout_risk:available_to_sell<=0_or_on_hand<=safety_stock")
    elif available <= float(cfg.low_stock_available_units):
        stock_status = "low_stock"
        risk_flag = True
        rules.append("low_stock:available_to_sell<=low_stock_available_units")
    else:
        # Overstock heuristics: high on-hand combined with weak demand proxy and/or high returns.
        if on_hand >= float(cfg.overstock_on_hand_units) and (
            mean_daily_revenue <= float(cfg.low_revenue_usd_per_day) or total_returns >= float(cfg.high_returns_units)
        ):
            stock_status = "overstocked"
            risk_flag = True
            if on_hand >= float(cfg.overstock_on_hand_units):
                rules.append("overstocked:on_hand>=overstock_on_hand_units")
            if mean_daily_revenue <= float(cfg.low_revenue_usd_per_day):
                rules.append("overstocked:mean_daily_revenue<=low_revenue_usd_per_day")
            if total_returns >= float(cfg.high_returns_units):
                rules.append("overstocked:total_returns>=high_returns_units")
        else:
            stock_status = "healthy"
            rules.append("healthy:default")

    return {
        "stock_status": stock_status,
        "risk_flag": bool(risk_flag),
        "signals": {
            "on_hand_units": float(on_hand),
            "safety_stock_units": float(safety),
            "available_to_sell": float(available),
            "mean_daily_revenue": float(mean_daily_revenue),
            "total_returns": float(total_returns),
        },
        "rules_fired": rules,
    }


def classify_inventory_for_candidates(
    candidates: list[dict[str, Any]],
    *,
    cfg: InventoryAgentConfig | None = None,
) -> dict[str, dict[str, Any]]:
    cfg2 = cfg or InventoryAgentConfig()
    out: dict[str, dict[str, Any]] = {}
    for c in candidates or []:
        if not isinstance(c, dict):
            continue
        pid = str(c.get("product_id") or "").strip()
        if not pid:
            continue
        out[pid] = classify_inventory_for_row(c, cfg=cfg2)
    return out


def attach_inventory_to_ranked_actions(
    ranked_actions: list[dict[str, Any]],
    *,
    cfg: InventoryAgentConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """
    Returns:
    - new_actions: same list with an `inventory` object attached per action
    - inventory_by_pid: mapping for trace/debug/eval reuse
    """
    inv_by_pid = classify_inventory_for_candidates(ranked_actions, cfg=cfg)
    new_actions: list[dict[str, Any]] = []
    for a in ranked_actions or []:
        if not isinstance(a, dict):
            continue
        pid = str(a.get("product_id") or "").strip()
        aa = dict(a)
        if pid and pid in inv_by_pid:
            aa["inventory"] = inv_by_pid[pid]
        else:
            aa["inventory"] = {"stock_status": "unknown", "risk_flag": False, "signals": {}, "rules_fired": ["unknown:no_product_id"]}
        new_actions.append(aa)
    return new_actions, inv_by_pid

