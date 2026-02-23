-- Replace project and dataset placeholders before execution.
-- Example: my-project-id.cod_fresh_start

CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.products` AS
SELECT
  SAFE_CAST(parent_asin AS STRING) AS product_id,
  ANY_VALUE(SAFE_CAST(title AS STRING)) AS title,
  ANY_VALUE(SAFE_CAST(text AS STRING)) AS description,
  ANY_VALUE(SAFE_CAST(rating AS FLOAT64)) AS avg_rating
FROM `{PROJECT_ID}.{DATASET}.stg_amazon_reviews`
WHERE parent_asin IS NOT NULL
GROUP BY parent_asin;


CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.reviews` AS
SELECT
  SAFE_CAST(parent_asin AS STRING) AS product_id,
  SAFE_CAST(user_id AS STRING) AS user_id,
  SAFE_CAST(rating AS FLOAT64) AS rating,
  SAFE_CAST(timestamp AS INT64) AS event_ts,
  SAFE_CAST(title AS STRING) AS title,
  SAFE_CAST(text AS STRING) AS review_text
FROM `{PROJECT_ID}.{DATASET}.stg_amazon_reviews`
WHERE parent_asin IS NOT NULL
  AND text IS NOT NULL;

