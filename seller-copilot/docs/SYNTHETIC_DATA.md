# Synthetic store & operations data

## Why this exists

The McAuley Amazon Reviews 2023 extract gives **reviews + product text + list price**, but a **business copilot** also needs plausible **inventory, unit economics, sales history, and supplier constraints** for recommendations, forecasting demos, and tabular models.

This layer is **synthetic**, **seeded** (reproducible), and **joined on `product_id`** to your curated `data/agent_dataset/products.parquet`.

## How to generate

```bash
python seller-copilot/scripts/generate_ops_data.py --seed 42 --days 180
```

Outputs go to `seller-copilot/data/synthetic/`:

| File | Contents |
|------|----------|
| `inventory_skus.parquet` | `unit_cost`, `margin_pct`, `stock_on_hand`, `reorder_point_units`, ABC class, shelf zone |
| `suppliers.parquet` | Supplier master (lead time, reliability) |
| `product_supplier_map.parquet` | `product_id` → `supplier_id`, MOQ |
| `sales_daily.parquet` | Long-format daily `units_sold`, `gross_revenue_usd` (sparse: only days with sales) |
| `marketing_spend_daily.parquet` | Sparse ad spend + channel + impressions proxy |
| `store_kpis_weekly.parquet` | Weekly revenue, estimated COGS, inventory value at cost |
| `synthetic_store_manifest.json` | Row counts, seed, file descriptions |

## Modeling notes

- **Margins** are drawn as a random COGS ratio × list price (or imputed list price when Amazon price is missing).
- **Demand** scales weakly with `review_count` (proxy for popularity), plus noise and seasonality.
- **Do not** claim these numbers are real Amazon seller metrics — label them **synthetic** in any report.

## Joins

```text
agent_dataset/products.parquet.product_id
  = inventory_skus.product_id
  = sales_daily.product_id
  = marketing_spend_daily.product_id
```
