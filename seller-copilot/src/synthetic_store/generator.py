"""
Generate realistic *store-like* synthetic data joined to ``product_id`` from the curated agent dataset.

Amazon reviews/meta do not include COGS, inventory, or true sales. These tables are for:
- tabular / forecasting models (demand, margin, stock risk),
- agent context (inventory-aware recommendations),
- demo KPIs without claiming they came from Amazon.

All values are reproducible via ``random_seed``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SyntheticStoreConfig:
    random_seed: int = 42
    sales_history_days: int = 180
    n_suppliers: int = 24
    start_date: str | None = None  # ISO date; default = today - sales_history_days


def _safe_price(p: pd.Series) -> np.ndarray:
    x = pd.to_numeric(p, errors="coerce").to_numpy(dtype=float)
    x = np.where(np.isfinite(x) & (x > 0), x, np.nan)
    return x


def _demand_scale(review_count: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Higher review volume → slightly higher baseline demand (noisy proxy)."""
    base = 0.35 + 0.65 * (np.log1p(np.maximum(review_count, 0.0)) / np.log1p(500.0))
    base = np.clip(base, 0.2, 3.0)
    return base * rng.lognormal(0.0, 0.25, size=base.shape)


def generate_inventory_skus(products: pd.DataFrame, cfg: SyntheticStoreConfig, rng: np.random.Generator) -> pd.DataFrame:
    """One row per product: inventory, costs, margin, ABC class."""
    n = len(products)
    pid = products["product_id"].astype(str).values
    ref_price = _safe_price(products["price"])
    review_count = pd.to_numeric(products.get("review_count", pd.Series(np.nan)), errors="coerce").fillna(0).to_numpy()

    # Impute a plausible list price when missing (Home & Kitchen wide range)
    synth_list = rng.lognormal(mean=np.log(28.0), sigma=1.1, size=n)
    list_price = np.where(np.isfinite(ref_price), ref_price, synth_list)

    # COGS as fraction of list price (typical retail gross margin band)
    cost_ratio = rng.uniform(0.38, 0.72, size=n)
    unit_cost = np.clip(list_price * cost_ratio, 0.5, None)
    margin_pct = np.clip((list_price - unit_cost) / np.maximum(list_price, 1e-6), -0.5, 0.95)

    # Stock levels: demand proxy × random shelf depth
    scale = _demand_scale(review_count, rng)
    mean_stock = np.clip(scale * rng.uniform(40, 220, size=n), 5, 50000)
    stock_on_hand = rng.poisson(np.clip(mean_stock, 1, 50000)).astype(int)
    reserved = rng.binomial(stock_on_hand, rng.uniform(0.0, 0.12, size=n)).astype(int)
    available = np.maximum(stock_on_hand - reserved, 0)

    reorder_point = np.maximum((scale * rng.uniform(8, 35, size=n)).astype(int), 3)
    safety_stock_days = rng.integers(7, 45, size=n)
    abc = rng.choice(["A", "B", "C"], size=n, p=[0.15, 0.35, 0.50])

    shelf = rng.choice(["A1", "A2", "B1", "B2", "C1", "DOCK"], size=n)

    return pd.DataFrame(
        {
            "sku_id": [f"SKU-{i:08d}" for i in range(n)],
            "product_id": pid,
            "reference_list_price": list_price,
            "unit_cost": unit_cost,
            "margin_pct": margin_pct,
            "stock_on_hand": stock_on_hand,
            "reserved_for_fulfillment": reserved,
            "available_to_sell": available,
            "reorder_point_units": reorder_point,
            "safety_stock_coverage_days": safety_stock_days,
            "abc_velocity_class": abc,
            "shelf_zone": shelf,
        }
    )


def generate_suppliers(cfg: SyntheticStoreConfig, rng: np.random.Generator) -> pd.DataFrame:
    names = [f"Supplier_{i+1:02d}_{rng.integers(1000, 9999)}" for i in range(cfg.n_suppliers)]
    lead = rng.integers(3, 28, size=cfg.n_suppliers)
    rel = rng.uniform(0.82, 0.995, size=cfg.n_suppliers)
    return pd.DataFrame(
        {
            "supplier_id": [f"SUP-{i:04d}" for i in range(cfg.n_suppliers)],
            "supplier_name": names,
            "typical_lead_time_days": lead,
            "reliability_score": rel,
            "payment_terms_days": rng.choice([15, 30, 30, 45, 60], size=cfg.n_suppliers),
        }
    )


def assign_suppliers(product_ids: np.ndarray, suppliers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    sids = suppliers["supplier_id"].values
    weights = rng.uniform(0.5, 1.5, size=len(sids))
    weights /= weights.sum()
    choice = rng.choice(sids, size=len(product_ids), p=weights)
    moq = rng.poisson(rng.uniform(12, 80, size=len(product_ids))).astype(int) + 1
    return pd.DataFrame({"product_id": product_ids, "supplier_id": choice, "minimum_order_qty": moq})


def generate_sales_daily(
    products: pd.DataFrame,
    inv: pd.DataFrame,
    cfg: SyntheticStoreConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Daily units and revenue per product (synthetic POS). Vectorized + sparse long format."""
    if cfg.start_date:
        start = date.fromisoformat(cfg.start_date)
    else:
        start = date.today() - timedelta(days=cfg.sales_history_days)

    pid = products["product_id"].astype(str).values
    n = len(pid)
    ddays = cfg.sales_history_days
    list_price = inv["reference_list_price"].to_numpy(dtype=float)
    review_count = pd.to_numeric(products.get("review_count", pd.Series(0)), errors="coerce").fillna(0).to_numpy()
    scale = _demand_scale(review_count, rng)

    t0 = np.arange(ddays)
    seasonal = 1.0 + 0.12 * np.sin(2 * np.pi * t0 / 7.0) + 0.08 * np.sin(2 * np.pi * t0 / 365.0 * 7)
    noise = rng.lognormal(0.0, 0.18, size=(n, ddays))
    lam = np.clip(scale[:, None] * 0.4 * seasonal[None, :] * noise, 0.05, None)
    units = rng.poisson(lam)
    cap = inv["available_to_sell"].to_numpy() // max(1, ddays // 30)
    cap = np.maximum(cap, rng.poisson(2, size=n))
    units = np.minimum(units, cap[:, None])
    disc = rng.uniform(0.88, 1.0, size=(n, ddays))
    revenue = units * list_price[:, None] * disc

    # Long format: only rows with activity (keeps file smaller)
    mask = units > 0
    ii, jj = np.where(mask)
    if len(ii) == 0:
        return pd.DataFrame(columns=["sale_date", "product_id", "units_sold", "gross_revenue_usd", "return_units"])

    day_str = np.array([((start + timedelta(days=int(j))).isoformat()) for j in jj])
    ret = rng.binomial(units[ii, jj], rng.uniform(0.0, 0.06, size=len(ii)))
    return pd.DataFrame(
        {
            "sale_date": day_str,
            "product_id": pid[ii],
            "units_sold": units[ii, jj].astype(int),
            "gross_revenue_usd": revenue[ii, jj].astype(float),
            "return_units": ret.astype(int),
        }
    )


def generate_marketing_daily(
    products: pd.DataFrame,
    cfg: SyntheticStoreConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sparse daily ad spend (not every product every day). Vectorized."""
    if cfg.start_date:
        start = date.fromisoformat(cfg.start_date)
    else:
        start = date.today() - timedelta(days=cfg.sales_history_days)

    pid = products["product_id"].astype(str).values
    n = len(pid)
    ddays = cfg.sales_history_days
    touch = rng.random((n, ddays)) < 0.42
    ii, jj = np.where(touch)
    if len(ii) == 0:
        return pd.DataFrame(columns=["date", "product_id", "ad_spend_usd", "channel", "impressions_proxy"])

    spend = rng.lognormal(mean=np.log(3.5), sigma=1.1, size=len(ii))
    spend = np.clip(spend, 0.25, 500.0)
    channels = rng.choice(["Sponsored_Products", "DSP", "Social", "Search"], size=len(ii), p=[0.45, 0.15, 0.2, 0.2])
    day_str = np.array([((start + timedelta(days=int(j))).isoformat()) for j in jj])
    return pd.DataFrame(
        {
            "date": day_str,
            "product_id": pid[ii],
            "ad_spend_usd": spend.astype(float),
            "channel": channels,
            "impressions_proxy": rng.poisson(200 + spend * 40).astype(int),
        }
    )


def generate_store_kpis_weekly(sales: pd.DataFrame, inv: pd.DataFrame, cfg: SyntheticStoreConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Weekly rollups for dashboard / agent context."""
    if sales.empty:
        return pd.DataFrame(columns=["week_start", "total_revenue_usd", "total_units", "estimated_cogs_usd", "inventory_value_at_cost"])

    s = sales.copy()
    s["sale_date"] = pd.to_datetime(s["sale_date"], utc=False)
    cost_map = inv.set_index("product_id")["unit_cost"]
    s["cogs_line"] = s["units_sold"].astype(float) * s["product_id"].map(cost_map).astype(float)

    g = s.groupby(pd.Grouper(key="sale_date", freq="W-MON"), as_index=False).agg(
        total_revenue_usd=("gross_revenue_usd", "sum"),
        total_units=("units_sold", "sum"),
        estimated_cogs_usd=("cogs_line", "sum"),
    )
    g.rename(columns={"sale_date": "week_start"}, inplace=True)
    g["week_start"] = g["week_start"].dt.strftime("%Y-%m-%d")

    inv_val = float((inv["unit_cost"] * inv["stock_on_hand"]).sum())
    g["inventory_value_at_cost"] = inv_val * rng.uniform(0.94, 1.06, size=len(g))
    return g


def generate_all(
    products_parquet: Path,
    output_dir: Path,
    cfg: SyntheticStoreConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or SyntheticStoreConfig()
    rng = np.random.default_rng(cfg.random_seed)

    products = pd.read_parquet(products_parquet)
    if "product_id" not in products.columns:
        raise ValueError("products parquet must contain product_id")
    products = products.drop_duplicates("product_id").reset_index(drop=True)

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    inv = generate_inventory_skus(products, cfg, rng)
    suppliers = generate_suppliers(cfg, rng)
    sku_supplier = assign_suppliers(inv["product_id"].values, suppliers, rng)
    sales = generate_sales_daily(products, inv, cfg, rng)
    marketing = generate_marketing_daily(products, cfg, rng)
    kpis = generate_store_kpis_weekly(sales, inv, cfg, rng)

    inv.to_parquet(output_dir / "inventory_skus.parquet", index=False)
    suppliers.to_parquet(output_dir / "suppliers.parquet", index=False)
    sku_supplier.to_parquet(output_dir / "product_supplier_map.parquet", index=False)
    sales.to_parquet(output_dir / "sales_daily.parquet", index=False)
    marketing.to_parquet(output_dir / "marketing_spend_daily.parquet", index=False)
    kpis.to_parquet(output_dir / "store_kpis_weekly.parquet", index=False)

    manifest = {
        "schema_version": 1,
        "purpose": "Operational projections keyed to product catalog (inventory, sales, marketing).",
        "random_seed": cfg.random_seed,
        "sales_history_days": cfg.sales_history_days,
        "n_products": len(products),
        "row_counts": {
            "inventory_skus": len(inv),
            "suppliers": len(suppliers),
            "product_supplier_map": len(sku_supplier),
            "sales_daily": len(sales),
            "marketing_spend_daily": len(marketing),
            "store_kpis_weekly": len(kpis),
        },
        "files": {
            "inventory_skus.parquet": "SKU costs, stock, reorder params (join: product_id)",
            "suppliers.parquet": "Supplier master",
            "product_supplier_map.parquet": "product_id → supplier_id, MOQ",
            "sales_daily.parquet": "Daily units/revenue (long)",
            "marketing_spend_daily.parquet": "Daily ad spend by product (sparse)",
            "store_kpis_weekly.parquet": "Weekly revenue/COGS-style rollups",
        },
        "join_key": "product_id (matches agent_dataset/products.parquet)",
        "config": asdict(cfg),
    }
    (output_dir / "synthetic_store_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
