# -*- coding: utf-8 -*-
"""
Generate sample outputs for Report Sections 3.2-3.6.

Run from repo root:
    .venv-copilot-v2\\Scripts\\python.exe copilot-v2\\src\\copilot_v2\\scripts\\utils\\generate_section_samples.py

Outputs:
    - Console tables for sections 3.2-3.5
    - Five PNG charts saved to copilot-v2/docs/figures/ for section 3.6
"""

import json
import os
import sys
from pathlib import Path

# Force UTF-8 output so Unicode chars render on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[5]
SNAP = REPO / "copilot-v2/artifacts/data_snapshots/38710839ca6e1009"
FIGURES = REPO / "copilot-v2/docs/figures"
FIGURES.mkdir(exist_ok=True)

SPLITS = REPO / "copilot-v2/artifacts/splits/38710839ca6e1009"

MANIFEST        = SNAP   / "manifest.json"
PRODUCTS        = SNAP   / "products.parquet"
REVIEWS         = SNAP   / "reviews.parquet"
SIGNALS         = SNAP   / "product_signals.parquet"
CORPUS          = SNAP   / "retrieval_corpus.parquet"
BALANCED        = SNAP   / "reviews_sentiment_balanced.parquet"
PRODUCTS_SPLIT  = SPLITS / "products_split.parquet"
REVIEWS_SPLIT   = SPLITS / "reviews_split.parquet"
SPLIT_CONFIG    = SPLITS / "split_config.json"

DISPLAY_COLS = {
    "products": ["product_id", "title", "brand", "subcategory", "price",
                 "avg_rating", "rating_count", "review_count"],
    "reviews":  ["review_id", "product_id", "review_title", "rating",
                 "helpful_vote", "event_ts"],
    "signals":  ["product_id", "review_count", "positive_ratio",
                 "avg_star_rating", "recent_review_ratio_90d",
                 "days_since_last_review", "price_percentile_in_subcategory",
                 "rating_vs_subcategory_mean"],
    "corpus":   ["product_id", "subcategory", "price", "avg_rating",
                 "product_document"],
}

SEED = 42
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 200)


def sep(title: str) -> None:
    bar = "=" * 80
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


# ── Load data (lazy cache) ────────────────────────────────────────────────────
print("Loading parquet files ...")
products        = pd.read_parquet(PRODUCTS)
reviews         = pd.read_parquet(REVIEWS)
signals         = pd.read_parquet(SIGNALS)
corpus          = pd.read_parquet(CORPUS)
balanced        = pd.read_parquet(BALANCED)
products_split  = pd.read_parquet(PRODUCTS_SPLIT)
reviews_split   = pd.read_parquet(REVIEWS_SPLIT)

with open(MANIFEST) as f:
    manifest = json.load(f)

print("Done.\n")


# ══════════════════════════════════════════════════════════════════════════════
# S. 3.2  Data Collection
# ══════════════════════════════════════════════════════════════════════════════
sep("S. 3.2  Data Collection - Raw Dataset Samples")

print("Collection constraints (from manifest.json):")
qg = manifest["quality_gates"]
sub = manifest["subset"]
print(f"  Source          : Amazon Reviews 2023 - Home & Kitchen")
print(f"  Min reviews/product : {qg['min_reviews_per_product']}")
print(f"  Review year floor   : {qg['recent_year_floor']}")
print(f"  Min review chars    : {qg['min_review_chars']}")
print(f"  Min title chars     : {qg['min_title_chars']}")
print(f"  Price range (USD)   : {qg['price_bounds']['low']} - {qg['price_bounds']['high']}")
print(f"  Max products        : {sub['max_products']:,}")
print(f"  Max per subcategory : {sub['max_products_per_subcategory']:,}")

print(f"\nPost-collection row counts:")
rc = manifest["row_counts"]
print(f"  products.parquet        : {rc['products']:>10,} rows")
print(f"  reviews.parquet         : {rc['reviews']:>10,} rows")

print("\nSample - reviews (5 rows):")
rv_sample = reviews[DISPLAY_COLS["reviews"]].sample(5, random_state=SEED)
print(rv_sample.to_string(index=False))

print("\nSample - products (5 rows):")
pr_sample = products[DISPLAY_COLS["products"]].dropna(subset=["price"]).sample(
    5, random_state=SEED
)
print(pr_sample.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# S. 3.3  Data Pre-processing
# ══════════════════════════════════════════════════════════════════════════════
sep("S. 3.3  Data Pre-processing - Preprocessed Dataset Samples")

print("Cleaning rules applied:")
print("  - Remove records with no valid product_id (asin / parent_asin)")
print("  - Parse timestamps -> UTC")
print("  - Cap review text at 5,000 chars; descriptions at 15,000 chars")
print("  - Remove non-numeric / negative prices (set to None)")
print("  - Winsorize prices to 1st-99th percentile")
print("  - Inner-join reviews <-> product metadata")

print("\nPost-cleaning statistics:")
print(f"  reviews.parquet  : {len(reviews):>10,} rows × {reviews.shape[1]} cols")
print(f"  products.parquet : {len(products):>10,} rows × {products.shape[1]} cols")

print(f"\n  Rating distribution (reviews):")
for rating, cnt in reviews["rating"].value_counts().sort_index().items():
    bar = "#" * int(cnt / 200_000)
    print(f"    {rating:.0f}*  {cnt:>8,}  {bar}")

print(f"\n  Price stats (products - winsorized USD):")
print(products["price"].describe().rename(lambda x: f"    {x}").to_string())

print("\nSample - cleaned products (5 rows):")
print(
    products[DISPLAY_COLS["products"]]
    .dropna(subset=["price"])
    .sample(5, random_state=SEED + 1)
    .to_string(index=False)
)

print("\nSample - cleaned reviews (5 rows):")
print(reviews[DISPLAY_COLS["reviews"]].sample(5, random_state=SEED + 1).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# S. 3.4  Data Transformation
# ══════════════════════════════════════════════════════════════════════════════
sep("S. 3.4  Data Transformation - Transformed Dataset Samples")

print("Transformation steps:")
print("  1. JSONL -> Parquet (columnar storage, faster I/O)")
print("  2. Review-level -> product-level aggregation (product_signals.parquet)")
print("  3. Subcategory metrics: price percentile, mean/median, rating delta")
print("  4. Text retrieval representation (retrieval_corpus.parquet)")

print(f"\nTransformed datasets:")
print(f"  products.parquet         : {len(products):>8,} rows × {products.shape[1]} cols")
print(f"  reviews.parquet          : {len(reviews):>8,} rows × {reviews.shape[1]} cols")
print(f"  product_signals.parquet  : {len(signals):>8,} rows × {signals.shape[1]} cols")
print(f"  retrieval_corpus.parquet : {len(corpus):>8,} rows × {corpus.shape[1]} cols")

print("\nSample - product_signals.parquet (5 rows):")
print(signals[DISPLAY_COLS["signals"]].sample(5, random_state=SEED).to_string(index=False))

print("\nSample - retrieval_corpus.parquet (3 rows, document truncated):")
corp_sample = corpus[DISPLAY_COLS["corpus"]].sample(3, random_state=SEED).copy()
corp_sample["product_document"] = corp_sample["product_document"].str[:120] + "..."
print(corp_sample.to_string(index=False))

print("\nFull product_document example (1 record):")
doc = corpus["product_document"].sample(1, random_state=SEED).iloc[0]
print(doc[:600])


# ══════════════════════════════════════════════════════════════════════════════
# S. 3.5  Data Preparation (Train / Val / Test splits)
# ══════════════════════════════════════════════════════════════════════════════
sep("S. 3.5  Data Preparation - Train / Val / Test Split Samples")

with open(SPLIT_CONFIG) as f:
    split_cfg = json.load(f)
print(f"Split strategy: time-based ({split_cfg.get('strategy', '')}), seed={split_cfg.get('random_seed', 42)}, ratio 70/15/15")

# Join split labels back to full feature dataframes
products_labeled = products.merge(products_split[["product_id", "split"]], on="product_id")
reviews_labeled  = reviews.merge(reviews_split[["review_id", "split"]], on="review_id")

products_train = products_labeled[products_labeled["split"] == "train"]
products_val   = products_labeled[products_labeled["split"] == "val"]
products_test  = products_labeled[products_labeled["split"] == "test"]

reviews_train  = reviews_labeled[reviews_labeled["split"] == "train"]
reviews_val    = reviews_labeled[reviews_labeled["split"] == "val"]
reviews_test   = reviews_labeled[reviews_labeled["split"] == "test"]

print("\nSplit sizes:")
header = f"  {'Split':<10} {'Products':>10} {'Reviews':>12}"
print(header)
print("  " + "-" * (len(header) - 2))
for label, pr, rv in [
    ("Train (70%)", products_train, reviews_train),
    ("Val   (15%)", products_val,   reviews_val),
    ("Test  (15%)", products_test,  reviews_test),
]:
    print(f"  {label:<10} {len(pr):>10,} {len(rv):>12,}")
print(f"  {'TOTAL':<10} {len(products_split):>10,} {len(reviews_split):>12,}")

print("\nSample - products_train (5 rows):")
print(
    products_train[DISPLAY_COLS["products"]]
    .dropna(subset=["price"])
    .sample(5, random_state=SEED)
    .to_string(index=False)
)

print("\nSample - reviews_val (5 rows):")
print(
    reviews_val[DISPLAY_COLS["reviews"]]
    .sample(5, random_state=SEED)
    .to_string(index=False)
)


# ══════════════════════════════════════════════════════════════════════════════
# S. 3.6  Data Statistics - Visualizations
# ══════════════════════════════════════════════════════════════════════════════
sep("S. 3.6  Data Statistics - Visualizations (saving PNGs to docs/figures/)")

BLUE   = "#4C72B0"
GREEN  = "#55A868"
ORANGE = "#DD8452"
RED    = "#C44E52"
PURPLE = "#8172B2"

# ── Fig 1: Dataset Size by Stage (log scale, all outputs in one view) ─────────
dataset_labels = [
    "products\n.parquet",
    "product_signals\n.parquet",
    "retrieval_corpus\n.parquet",
    "reviews\n.parquet",
    "reviews_sentiment\n_balanced.parquet",
]
dataset_counts = [len(products), len(signals), len(corpus), len(reviews), len(balanced)]
dataset_colors = [BLUE, BLUE, BLUE, GREEN, GREEN]
dataset_groups = ["Product-scale (50k)", "Product-scale (50k)", "Product-scale (50k)",
                  "Review-scale (5M+)", "Review-scale (5M+)"]

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(dataset_labels, dataset_counts, color=dataset_colors, edgecolor="white")
ax.set_yscale("log")
ax.set_ylabel("Row count (log scale)")
ax.set_title("Dataset Size by Stage (all curated outputs)", fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mtick.FuncFormatter(
    lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x:,.0f}"
))

ax.set_ylim(top=ax.get_ylim()[1] * 5)  # headroom so labels don't clip title
for bar, v in zip(bars, dataset_counts):
    label = f"{v/1e6:.2f}M" if v >= 1e6 else f"{v:,}"
    ax.text(bar.get_x() + bar.get_width() / 2, v * 1.3, label,
            ha="center", va="bottom", fontsize=9)

from matplotlib.patches import Patch
legend_handles = [Patch(color=BLUE, label="Product-scale (~50k rows)"),
                  Patch(color=GREEN, label="Review-scale (~5M rows)")]
ax.legend(handles=legend_handles)
plt.tight_layout()
out = FIGURES / "fig1_dataset_size_by_stage.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out.name}")


# ── Fig 2: Train / Val / Test Split Distribution ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Train / Val / Test Split Distribution (70 / 15 / 15)", fontsize=13, fontweight="bold")

split_labels = ["Train\n(70%)", "Val\n(15%)", "Test\n(15%)"]
split_colors = [BLUE, GREEN, ORANGE]

split_products = [len(products_train), len(products_val), len(products_test)]
axes[0].bar(split_labels, split_products, color=split_colors)
axes[0].set_title("Products per Split")
axes[0].set_ylabel("Row count")
axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}"))
for i, v in enumerate(split_products):
    axes[0].text(i, v + 150, f"{v:,}", ha="center", fontsize=9)

split_reviews = [len(reviews_train), len(reviews_val), len(reviews_test)]
axes[1].bar(split_labels, split_reviews, color=split_colors)
axes[1].set_title("Reviews per Split")
axes[1].set_ylabel("Row count")
axes[1].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1e6:.2f}M"))
for i, v in enumerate(split_reviews):
    axes[1].text(i, v + 15_000, f"{v/1e6:.2f}M", ha="center", fontsize=9)

plt.tight_layout()
out = FIGURES / "fig2_split_distribution.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out.name}")


# ── Fig 3: Price Distribution (winsorized) ────────────────────────────────────
prices = products["price"].dropna()
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(prices, bins=60, color=BLUE, edgecolor="white", linewidth=0.4)
ax.axvline(prices.median(), color=RED, linestyle="--", linewidth=1.5,
           label=f"Median ${prices.median():.2f}")
ax.axvline(prices.mean(), color=ORANGE, linestyle="--", linewidth=1.5,
           label=f"Mean ${prices.mean():.2f}")
ax.set_title("Price Distribution (winsorized $5.75 - $424)", fontsize=13, fontweight="bold")
ax.set_xlabel("Price (USD)")
ax.set_ylabel("Product count")
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend()
out = FIGURES / "fig3_price_distribution.png"
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out.name}")


# ── Fig 4: Star Rating Distribution Before vs After Balancing ─────────────────
# Show individual star ratings so the actual trimming (4* and 5*) is visible
star_labels = ["1*", "2*", "3*", "4*", "5*"]
rv_vals  = [reviews["rating"].value_counts().get(float(r), 0) for r in [1,2,3,4,5]]
bal_vals = [balanced["rating"].value_counts().get(float(r), 0) for r in [1,2,3,4,5]]

x = np.arange(len(star_labels))
w = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - w / 2, rv_vals,  w, label="Before balancing", color=BLUE)
bars2 = ax.bar(x + w / 2, bal_vals, w, label="After balancing",  color=GREEN)
ax.set_title("Star Rating Distribution Before vs After Sentiment Balancing",
             fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(star_labels)
ax.set_xlabel("Star rating")
ax.set_ylabel("Review count")
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v/1e6:.2f}M"))
ax.legend()
for bar in [*bars1, *bars2]:
    h = bar.get_height()
    label = f"{h/1e6:.2f}M" if h >= 1e5 else f"{h:,}"
    ax.text(bar.get_x() + bar.get_width() / 2, h + 8_000,
            label, ha="center", va="bottom", fontsize=8)
plt.tight_layout()
out = FIGURES / "fig4_sentiment_distribution.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out.name}")


# ── Fig 5: Histogram of subcategory sizes ────────────────────────────────────
# Shows how products are distributed across subcategories (distinct from 3.7.6
# which is a bar chart of top subcategories; this shows the overall distribution)
sub_counts = products["subcategory"].value_counts()
capped  = sub_counts[sub_counts == 1000]
uncapped = sub_counts[sub_counts < 1000]

bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
bin_labels = ["1-100","101-200","201-300","301-400","401-500",
              "501-600","601-700","701-800","801-900","901-999"]
bin_vals = pd.cut(uncapped, bins=bins, labels=bin_labels, right=True).value_counts().reindex(bin_labels)

all_labels = bin_labels + ["1,000\n(capped)"]
all_vals   = list(bin_vals.values) + [len(capped)]
colors = [PURPLE] * len(bin_labels) + [RED]
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(all_labels, all_vals, color=colors, edgecolor="white")
ax.set_title("Distribution of Products per Subcategory\n(how many subcategories fall in each size bucket)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Products per subcategory")
ax.set_ylabel("Number of subcategories")
for bar, v in zip(bars, all_vals):
    if v > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, str(v),
                ha="center", va="bottom", fontsize=9)

from matplotlib.patches import Patch
legend_handles = [Patch(color=PURPLE, label="Subcategories below cap"),
                  Patch(color=RED,    label="Subcategories at cap (1,000)")]
ax.legend(handles=legend_handles)
plt.tight_layout()
out = FIGURES / "fig5_subcategory_distribution.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out.name}")


# ── Done ──────────────────────────────────────────────────────────────────────
sep("Complete")
print(f"All figures saved to: {FIGURES}")
print("Copy the PNG files into your report for Section 3.6.\n")
