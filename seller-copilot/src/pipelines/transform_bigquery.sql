-- Canonical tables for seller copilot.
-- Runtime parameters: {PROJECT_ID}, {DATASET}

CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.products`
PARTITION BY DATE(ingested_at)
CLUSTER BY product_id AS
WITH review_agg AS (
  SELECT
    product_id,
    ANY_VALUE(NULLIF(review_title, "")) AS review_title,
    ANY_VALUE(NULLIF(review_text, "")) AS review_description,
    AVG(rating) AS avg_rating,
    APPROX_QUANTILES(price, 2)[OFFSET(1)] AS review_median_price,
    MAX(ingested_at) AS review_ingested_at
  FROM `{PROJECT_ID}.{DATASET}.stg_amazon_reviews`
  WHERE product_id IS NOT NULL AND product_id != ''
  GROUP BY product_id
),
meta_agg AS (
  SELECT
    product_id,
    ANY_VALUE(NULLIF(title, "")) AS meta_title,
    ANY_VALUE(NULLIF(description, "")) AS meta_description,
    APPROX_QUANTILES(price, 2)[OFFSET(1)] AS meta_median_price,
    MAX(ingested_at) AS meta_ingested_at
  FROM `{PROJECT_ID}.{DATASET}.stg_amazon_meta`
  WHERE product_id IS NOT NULL AND product_id != ''
  GROUP BY product_id
)
SELECT
  COALESCE(r.product_id, m.product_id) AS product_id,
  COALESCE(m.meta_title, r.review_title) AS title,
  COALESCE(m.meta_description, r.review_description) AS description,
  COALESCE(r.avg_rating, 0.0) AS avg_rating,
  COALESCE(r.review_median_price, m.meta_median_price) AS median_price,
  GREATEST(COALESCE(r.review_ingested_at, TIMESTAMP("1970-01-01")), COALESCE(m.meta_ingested_at, TIMESTAMP("1970-01-01"))) AS ingested_at
FROM review_agg r
FULL OUTER JOIN meta_agg m USING (product_id);


CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.reviews`
PARTITION BY DATE(ingested_at)
CLUSTER BY product_id, user_id AS
SELECT
  product_id,
  user_id,
  rating,
  review_title,
  review_text,
  event_ts,
  source,
  ingested_at
FROM `{PROJECT_ID}.{DATASET}.stg_amazon_reviews`
WHERE product_id IS NOT NULL
  AND review_text IS NOT NULL
  AND TRIM(review_text) != '';


CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.support_tickets`
PARTITION BY DATE(ingested_at)
CLUSTER BY ticket_id, user_id AS
SELECT
  ticket_id,
  user_id,
  text AS ticket_text,
  label,
  created_at,
  source,
  ingested_at
FROM `{PROJECT_ID}.{DATASET}.stg_twitter_support`
WHERE ticket_id IS NOT NULL
  AND text IS NOT NULL
  AND TRIM(text) != '';


CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.retail_transactions`
PARTITION BY DATE(ingested_at)
CLUSTER BY invoice_no, stock_code AS
SELECT
  invoice_no,
  stock_code,
  description,
  quantity,
  invoice_ts,
  unit_price,
  customer_id,
  country,
  source,
  ingested_at
FROM `{PROJECT_ID}.{DATASET}.stg_online_retail`
WHERE invoice_no IS NOT NULL;


CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.churn_signals`
PARTITION BY DATE(ingested_at)
CLUSTER BY customer_id AS
SELECT
  customer_id,
  gender,
  senior_citizen,
  tenure,
  monthly_charges,
  total_charges,
  churn_label,
  source,
  ingested_at
FROM `{PROJECT_ID}.{DATASET}.stg_telco_churn`
WHERE customer_id IS NOT NULL;
