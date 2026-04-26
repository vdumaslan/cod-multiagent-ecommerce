from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _lognormal_params_from_price_tier(median_price: float, n_skus: int) -> tuple[float, float]:
    """
    Heuristic calibration:
    - Higher price tiers => lower typical on-hand (capital intensive)
    - Larger catalogs => wider dispersion
    """
    med = float(median_price) if np.isfinite(median_price) else 25.0
    n = max(int(n_skus), 1)

    # Map median price into a rough tier multiplier.
    tier = np.clip(np.log1p(med) / np.log1p(200.0), 0.0, 1.0)  # 0..1
    base_mu_units = 4.4 - 1.2 * tier  # ln(units)
    base_sigma = 0.55 + 0.25 * np.clip(np.log10(n + 10) / 3.0, 0.0, 1.0)
    return float(base_mu_units), float(base_sigma)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-id", required=True)
    ap.add_argument("--artifacts-root", type=Path, default=Path("copilot-v2/artifacts"))
    ap.add_argument("--n-sale-days", type=int, default=90)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start-date", default="", help="Optional YYYY-MM-DD (default: today - n_sale_days).")
    args = ap.parse_args()

    if args.n_sale_days <= 0:
        raise SystemExit("--n-sale-days must be > 0")

    repo_root = Path(__file__).resolve().parents[4]
    artifacts_root = (repo_root / args.artifacts_root).resolve() if not args.artifacts_root.is_absolute() else args.artifacts_root
    snap_dir = artifacts_root / "data_snapshots" / str(args.snapshot_id)
    products_path = snap_dir / "products.parquet"
    if not products_path.exists():
        raise SystemExit(f"Missing {products_path}")

    products = pd.read_parquet(products_path)
    if "product_id" not in products.columns or "subcategory" not in products.columns:
        raise SystemExit("products.parquet must include product_id and subcategory")

    price_col = "price_usd" if "price_usd" in products.columns else ("price" if "price" in products.columns else "")
    if not price_col:
        raise SystemExit("products.parquet must include price or price_usd for synthetic ops calibration")
    products[price_col] = pd.to_numeric(products[price_col], errors="coerce")

    rng = np.random.default_rng(args.seed)

    out_dir = artifacts_root / "synthetic" / str(args.snapshot_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inventory per SKU (one row per product_id).
    inv_rows: list[dict] = []
    sales_rows: list[dict] = []

    if args.start_date.strip():
        start = datetime.fromisoformat(args.start_date.strip()).replace(tzinfo=timezone.utc)
    else:
        start = datetime.now(timezone.utc) - timedelta(days=args.n_sale_days)
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)

    for subcat, grp in products.groupby("subcategory", dropna=False):
        g = grp.copy()
        med_price = float(np.nanmedian(g[price_col].to_numpy(dtype=float))) if len(g) else 25.0
        mu, sigma = _lognormal_params_from_price_tier(med_price, len(g))

        # On-hand inventory: lognormal -> integer units.
        on_hand = rng.lognormal(mean=mu, sigma=sigma, size=len(g))
        on_hand = np.clip(on_hand, 0, 5000)
        on_hand_units = np.round(on_hand).astype(int)

        # Safety stock as fraction of on-hand with noise (at least 0).
        frac = rng.uniform(0.10, 0.35, size=len(g))
        safety = np.floor(on_hand_units * frac).astype(int)
        safety = np.clip(safety, 0, on_hand_units)

        # Sales intensity: Poisson rate depends on subcategory tier.
        # Higher price -> lower units velocity. Keep non-zero via floor.
        tier = np.clip(np.log1p(med_price) / np.log1p(200.0), 0.0, 1.0)
        base_lambda = 1.2 - 0.7 * tier  # units/day
        base_lambda = float(np.clip(base_lambda, 0.15, 2.5))

        # Per-SKU latent multiplier.
        sku_mult = rng.lognormal(mean=0.0, sigma=0.65, size=len(g))
        sku_lambda = base_lambda * sku_mult

        # Price jitter around median (for revenue calc).
        price = g[price_col].to_numpy(dtype=float)
        price = np.where(np.isfinite(price), price, med_price)
        price_jitter = rng.uniform(0.90, 1.10, size=len(g))
        eff_price = np.clip(price * price_jitter, 1.0, None)

        # Returns: per-unit probability (higher for low-rated? we don't have rating here, so simple tier-based).
        ret_p = 0.03 + 0.04 * tier  # 3%..7%
        ret_p = float(np.clip(ret_p, 0.01, 0.20))

        for i, pid in enumerate(g["product_id"].astype(str).tolist()):
            oh = int(on_hand_units[i])
            ss = int(safety[i])
            inv_rows.append(
                {
                    "product_id": pid,
                    "subcategory": None if (isinstance(subcat, float) and np.isnan(subcat)) else str(subcat),
                    "on_hand_units": oh,
                    "safety_stock_units": ss,
                    "available_to_sell": int(max(0, oh - ss)),
                }
            )

            # Daily sales table for this SKU.
            lam = float(np.clip(sku_lambda[i], 0.01, 50.0))
            sold = rng.poisson(lam=lam, size=args.n_sale_days).astype(int)
            gross_rev = sold.astype(float) * float(eff_price[i]) * rng.uniform(0.95, 1.05, size=args.n_sale_days)
            gross_rev = np.round(gross_rev, 2)
            returns = rng.binomial(n=sold, p=ret_p, size=args.n_sale_days).astype(int)

            for d in range(args.n_sale_days):
                day = (start + timedelta(days=d)).date().isoformat()
                sales_rows.append(
                    {
                        "product_id": pid,
                        "date": day,
                        "units_sold": int(sold[d]),
                        "gross_revenue_usd": float(gross_rev[d]),
                        "return_units": int(returns[d]),
                    }
                )

    inv = pd.DataFrame(inv_rows)
    sales = pd.DataFrame(sales_rows)

    inv_path = out_dir / "inventory_skus.parquet"
    sales_path = out_dir / "sales_daily.parquet"
    inv.to_parquet(inv_path, index=False)
    sales.to_parquet(sales_path, index=False)

    meta = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": str(args.snapshot_id),
        "seed": int(args.seed),
        "n_sale_days": int(args.n_sale_days),
        "start_date_utc": start.isoformat(),
        "rows": {"inventory_skus": int(len(inv)), "sales_daily": int(len(sales))},
        "output_dir": str(out_dir),
    }
    (out_dir / "synthetic_ops_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps({"ok": True, "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()

