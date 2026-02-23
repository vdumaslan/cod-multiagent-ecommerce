-- Replace project and dataset placeholders before execution.

CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.product_features` AS
SELECT
  p.product_id,
  p.title,
  p.description,
  p.avg_rating,
  CONCAT(IFNULL(p.title, ""), " ", IFNULL(p.description, "")) AS product_document,
  COUNT(r.review_text) AS review_count,
  AVG(r.rating) AS review_avg_rating
FROM `{PROJECT_ID}.{DATASET}.products` p
LEFT JOIN `{PROJECT_ID}.{DATASET}.reviews` r
  ON p.product_id = r.product_id
GROUP BY 1,2,3,4,5;

