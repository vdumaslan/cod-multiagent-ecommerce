from __future__ import annotations

import pandas as pd


def _clean_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def normalize_meta(raw: pd.DataFrame) -> pd.DataFrame:
    title = _clean_text(raw.get("title", pd.Series(dtype="object")))
    product_id = raw.get("parent_asin", raw.get("asin", pd.Series(dtype="object"))).astype(str)
    category = _clean_text(raw.get("main_category", pd.Series(dtype="object")))
    brand = _clean_text(raw.get("store", raw.get("brand", pd.Series(dtype="object"))))
    description = raw.get("description", pd.Series(dtype="object")).fillna("")
    if description.apply(lambda x: isinstance(x, list)).any():
        description = description.apply(lambda x: " ".join(str(v) for v in x) if isinstance(x, list) else str(x))
    description = _clean_text(description)
    price = (
        raw.get("price", pd.Series(dtype="object"))
        .astype(str)
        .str.replace(r"[^0-9.]", "", regex=True)
        .replace("", pd.NA)
    )
    price = pd.to_numeric(price, errors="coerce")
    rating = pd.to_numeric(raw.get("average_rating", pd.Series(dtype="float64")), errors="coerce")
    rating_count = pd.to_numeric(raw.get("rating_number", pd.Series(dtype="float64")), errors="coerce")

    out = pd.DataFrame(
        {
            "product_id": product_id,
            "title": title,
            "brand": brand,
            "category": category.replace("", "Home_and_Kitchen"),
            "subcategory": category.replace("", "Home_and_Kitchen"),
            "description": description,
            "price": price,
            "avg_rating": rating,
            "rating_count": rating_count,
        }
    )
    out["product_document"] = (out["title"] + " " + out["description"]).str.strip()
    return out


def normalize_reviews(raw: pd.DataFrame) -> pd.DataFrame:
    product_id = raw.get("parent_asin", raw.get("asin", pd.Series(dtype="object"))).astype(str)
    review_id = raw.get("review_id", pd.Series(range(len(raw)))).astype(str)
    review_title = _clean_text(raw.get("title", pd.Series(dtype="object")))
    review_text = _clean_text(raw.get("text", raw.get("content", pd.Series(dtype="object"))))
    rating = pd.to_numeric(raw.get("rating", pd.Series(dtype="float64")), errors="coerce")
    helpful_vote = pd.to_numeric(raw.get("helpful_vote", pd.Series(dtype="float64")), errors="coerce").fillna(0)
    event_ts = pd.to_datetime(raw.get("timestamp", pd.Series(dtype="float64")), errors="coerce", unit="ms", utc=True)

    return pd.DataFrame(
        {
            "review_id": review_id,
            "product_id": product_id,
            "review_title": review_title,
            "review_text": review_text,
            "rating": rating,
            "helpful_vote": helpful_vote,
            "event_ts": event_ts,
        }
    )


def curate_products_and_reviews(
    products: pd.DataFrame,
    reviews: pd.DataFrame,
    *,
    min_reviews_per_product: int,
    min_title_chars: int,
    min_review_chars: int,
    min_price_percentile: float,
    max_price_percentile: float,
    max_products: int,
    recent_year_floor: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    products = products.copy()
    reviews = reviews.copy()

    report: dict = {
        "initial_products": int(len(products)),
        "initial_reviews": int(len(reviews)),
        "gates": [],
    }

    products = products[
        products["product_id"].str.len().gt(0)
        & products["title"].str.len().ge(min_title_chars)
        & products["product_document"].str.len().ge(min_title_chars)
    ]
    report["after_product_basic_filters"] = int(len(products))
    reviews = reviews[
        reviews["product_id"].str.len().gt(0)
        & reviews["review_text"].str.len().ge(min_review_chars)
        & reviews["rating"].between(1, 5, inclusive="both")
    ]
    report["after_review_basic_filters"] = int(len(reviews))
    reviews = reviews[reviews["event_ts"].dt.year.fillna(0).ge(recent_year_floor)]
    report["after_review_recency_filter"] = int(len(reviews))

    review_stats = (
        reviews.groupby("product_id", as_index=False)
        .agg(
            review_count=("review_id", "count"),
            avg_review_rating=("rating", "mean"),
            avg_helpful_vote=("helpful_vote", "mean"),
            last_review_ts=("event_ts", "max"),
        )
    )
    products = products.merge(review_stats, on="product_id", how="inner")
    products = products[products["review_count"] >= min_reviews_per_product]
    report["after_min_reviews_per_product"] = int(len(products))

    valid_price = products["price"].dropna()
    if not valid_price.empty:
        low = valid_price.quantile(min_price_percentile)
        high = valid_price.quantile(max_price_percentile)
        products = products[products["price"].between(low, high, inclusive="both") | products["price"].isna()]
        report["price_bounds"] = {"low": float(low), "high": float(high)}
    else:
        report["price_bounds"] = None

    products["quality_score"] = (
        products["review_count"].clip(upper=1000).pow(0.5)
        + products["avg_rating"].fillna(0) * 2
        + products["product_document"].str.len().clip(upper=600) / 100
    )
    products = products.sort_values("quality_score", ascending=False).drop_duplicates("product_id")
    products = products.head(max_products)
    report["after_quality_score_topk"] = int(len(products))

    selected_ids = set(products["product_id"].astype(str))
    reviews = reviews[reviews["product_id"].astype(str).isin(selected_ids)]
    report["final_reviews"] = int(len(reviews))

    product_features = (
        reviews.assign(pos=(reviews["rating"] >= 4).astype(int))
        .groupby("product_id", as_index=False)
        .agg(
            review_count=("review_id", "count"),
            positive_ratio=("pos", "mean"),
            sentiment_score=("rating", "mean"),
            recency_score=("event_ts", lambda s: float(s.notna().mean())),
        )
    )
    products = products.drop(columns=["quality_score"], errors="ignore")
    report["final_products"] = int(len(products))
    report["final_product_signals"] = int(len(product_features))
    return (
        products.reset_index(drop=True),
        reviews.reset_index(drop=True),
        product_features.reset_index(drop=True),
        report,
    )

