-- Feature tables for model-ready datasets.
-- Runtime parameters: {PROJECT_ID}, {DATASET}, {TRAIN_RATIO}, {VAL_RATIO}, {SEED}

CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.product_features`
PARTITION BY DATE(ingested_at)
CLUSTER BY product_id AS
WITH review_agg AS (
  SELECT
    product_id,
    COUNT(*) AS review_count,
    AVG(rating) AS review_avg_rating,
    AVG(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) AS positive_ratio
  FROM `{PROJECT_ID}.{DATASET}.reviews`
  GROUP BY product_id
),
retail_price AS (
  SELECT
    APPROX_QUANTILES(unit_price, 2)[OFFSET(1)] AS retail_median_price
  FROM `{PROJECT_ID}.{DATASET}.retail_transactions`
  WHERE unit_price IS NOT NULL
)
SELECT
  p.product_id,
  COALESCE(p.title, "unknown_title") AS title,
  COALESCE(p.description, "") AS description,
  COALESCE(p.avg_rating, 0.0) AS avg_rating,
  COALESCE(p.median_price, retail_price.retail_median_price, 0.0) AS price,
  CONCAT(COALESCE(p.title, ""), " ", COALESCE(p.description, "")) AS product_document,
  COALESCE(review_agg.review_count, 0) AS review_count,
  COALESCE(review_agg.review_avg_rating, 0.0) AS review_avg_rating,
  COALESCE(review_agg.positive_ratio, 0.0) AS positive_ratio,
  p.ingested_at
FROM `{PROJECT_ID}.{DATASET}.products` p
LEFT JOIN review_agg ON p.product_id = review_agg.product_id
CROSS JOIN retail_price;


CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.sentiment_dataset`
PARTITION BY DATE(ingested_at)
CLUSTER BY product_id AS
SELECT
  product_id,
  review_text AS text,
  CASE
    WHEN rating >= 4 THEN 2
    WHEN rating = 3 THEN 1
    ELSE 0
  END AS label,
  ingested_at
FROM `{PROJECT_ID}.{DATASET}.reviews`
WHERE rating IS NOT NULL
  AND review_text IS NOT NULL
  AND TRIM(review_text) != '';


CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.ranking_pairs`
PARTITION BY DATE(ingested_at)
CLUSTER BY product_id AS
WITH bounds AS (
  SELECT
    APPROX_QUANTILES(price, 10)[OFFSET(3)] AS p30,
    APPROX_QUANTILES(price, 10)[OFFSET(7)] AS p70,
    APPROX_QUANTILES(SAFE_DIVIDE(avg_rating, NULLIF(price, 0)), 10)[OFFSET(5)] AS median_value_score
  FROM `{PROJECT_ID}.{DATASET}.product_features`
  WHERE price IS NOT NULL AND price > 0
),
base AS (
  SELECT
    pf.product_id,
    pf.price,
    pf.avg_rating,
    pf.positive_ratio,
    SAFE_DIVIDE(pf.avg_rating, NULLIF(pf.price, 0)) AS value_score,
    pf.ingested_at,
    b.p30,
    b.p70,
    b.median_value_score
  FROM `{PROJECT_ID}.{DATASET}.product_features` pf
  CROSS JOIN bounds b
)
SELECT
  "best budget products" AS query_text,
  product_id,
  CASE WHEN price <= p30 AND avg_rating >= 4.0 AND positive_ratio >= 0.60 THEN 1 ELSE 0 END AS relevance_label,
  ingested_at
FROM base
UNION ALL
SELECT
  "top rated products" AS query_text,
  product_id,
  CASE WHEN avg_rating >= 4.40 AND positive_ratio >= 0.75 THEN 1 ELSE 0 END AS relevance_label,
  ingested_at
FROM base
UNION ALL
SELECT
  "value for money products" AS query_text,
  product_id,
  CASE WHEN value_score >= median_value_score AND avg_rating >= 4.0 THEN 1 ELSE 0 END AS relevance_label,
  ingested_at
FROM base
UNION ALL
SELECT
  "premium quality products" AS query_text,
  product_id,
  CASE WHEN price >= p70 AND avg_rating >= 4.3 AND positive_ratio >= 0.70 THEN 1 ELSE 0 END AS relevance_label,
  ingested_at
FROM base;


CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.pricing_features`
PARTITION BY DATE(ingested_at)
CLUSTER BY product_id AS
SELECT
  product_id,
  price,
  avg_rating,
  review_count,
  positive_ratio,
  SAFE_DIVIDE(avg_rating, NULLIF(price, 0)) AS rating_price_ratio,
  price AS target_price,
  ingested_at
FROM `{PROJECT_ID}.{DATASET}.product_features`
WHERE price IS NOT NULL;


CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.retrieval_corpus`
PARTITION BY DATE(ingested_at)
CLUSTER BY product_id AS
SELECT
  product_id,
  product_document,
  price,
  avg_rating,
  ingested_at
FROM `{PROJECT_ID}.{DATASET}.product_features`;


CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.training_splits`
PARTITION BY DATE(created_at)
CLUSTER BY dataset_name, split AS
WITH base AS (
  SELECT
    "sentiment_dataset" AS dataset_name,
    CAST(FARM_FINGERPRINT(CONCAT(product_id, text, CAST(label AS STRING), CAST(ingested_at AS STRING))) AS INT64) AS h,
    CURRENT_TIMESTAMP() AS created_at
  FROM `{PROJECT_ID}.{DATASET}.sentiment_dataset`
)
SELECT
  dataset_name,
  CASE
    WHEN ABS(MOD(h, 10000)) < CAST(10000 * {TRAIN_RATIO} AS INT64) THEN "train"
    WHEN ABS(MOD(h, 10000)) < CAST(10000 * ({TRAIN_RATIO} + {VAL_RATIO}) AS INT64) THEN "val"
    ELSE "test"
  END AS split,
  COUNT(*) AS row_count,
  created_at
FROM base
GROUP BY dataset_name, split, created_at;
