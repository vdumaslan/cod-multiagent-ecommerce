from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import yaml


@dataclass
class Stage1Config:
    project_id: str
    dataset: str
    location: str
    repo_id: str
    review_path: str
    meta_path: str
    use_local_files: bool
    local_reviews_path: str | None
    local_meta_path: str | None
    max_reviews: int
    max_meta: int
    min_reviews_per_product: int
    max_price_percentile: float
    min_price_percentile: float
    max_products: int
    min_title_chars: int
    min_review_chars: int
    recent_year_floor: int
    reset_bigquery_tables: bool
    service_account_key_path: str | None
    require_bigquery: bool
    agent_dataset_dir: str


def load_config(path: str) -> Stage1Config:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    project = cfg["project"]
    source = cfg["source"]
    q = cfg["quality_gates"]
    ingest = cfg["ingestion"]

    project_id = os.getenv("GCP_PROJECT_ID", project["project_id"])
    dataset = os.getenv("BIGQUERY_DATASET", project["dataset"])
    location = os.getenv("BIGQUERY_LOCATION", project["location"])
    service_account_key_path = os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS",
        cfg.get("project", {}).get("service_account_key_path"),
    )
    repo_id = os.getenv("AMAZON_REPO_ID", source["repo_id"])
    review_path = os.getenv("AMAZON_HOME_KITCHEN_REVIEW_PATH", source["home_kitchen_review_path"])
    meta_path = os.getenv("AMAZON_HOME_KITCHEN_META_PATH", source["home_kitchen_meta_path"])
    use_local_files = os.getenv(
        "STAGE1_USE_LOCAL_FILES",
        str(source.get("use_local_files", False)),
    ).lower() in ("1", "true", "yes")
    local_reviews_path = (os.getenv("STAGE1_LOCAL_REVIEWS_PATH") or source.get("local_reviews_path") or "").strip() or None
    local_meta_path = (os.getenv("STAGE1_LOCAL_META_PATH") or source.get("local_meta_path") or "").strip() or None
    ops = cfg.get("ops", {})
    agent_dataset_dir = (
        os.getenv("STAGE1_AGENT_DATASET_DIR") or ops.get("agent_dataset_dir") or "seller-copilot/data/agent_dataset"
    ).strip()
    return Stage1Config(
        project_id=project_id,
        dataset=dataset,
        location=location,
        repo_id=repo_id,
        review_path=review_path,
        meta_path=meta_path,
        use_local_files=use_local_files,
        local_reviews_path=local_reviews_path,
        local_meta_path=local_meta_path,
        max_reviews=int(os.getenv("STAGE1_MAX_REVIEWS", ingest["max_reviews"])),
        max_meta=int(os.getenv("STAGE1_MAX_META", ingest["max_meta"])),
        min_reviews_per_product=int(os.getenv("STAGE1_MIN_REVIEWS_PER_PRODUCT", q["min_reviews_per_product"])),
        max_price_percentile=float(os.getenv("STAGE1_MAX_PRICE_PERCENTILE", q["max_price_percentile"])),
        min_price_percentile=float(os.getenv("STAGE1_MIN_PRICE_PERCENTILE", q["min_price_percentile"])),
        max_products=int(os.getenv("STAGE1_MAX_PRODUCTS", q["max_products"])),
        min_title_chars=int(os.getenv("STAGE1_MIN_TITLE_CHARS", q["min_title_chars"])),
        min_review_chars=int(os.getenv("STAGE1_MIN_REVIEW_CHARS", q["min_review_chars"])),
        recent_year_floor=int(os.getenv("STAGE1_RECENT_YEAR_FLOOR", q["recent_year_floor"])),
        reset_bigquery_tables=os.getenv(
            "STAGE1_RESET_BIGQUERY_TABLES",
            str(ops.get("reset_bigquery_tables", False)),
        ).lower()
        == "true",
        service_account_key_path=service_account_key_path,
        require_bigquery=os.getenv(
            "STAGE1_REQUIRE_BIGQUERY",
            str(ops.get("require_bigquery", True)),
        ).lower()
        == "true",
        agent_dataset_dir=agent_dataset_dir,
    )

