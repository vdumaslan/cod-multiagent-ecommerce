from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_pipeline_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise RuntimeError("Invalid pipeline config format.")
    return cfg


def get_project_settings(cfg: dict[str, Any]) -> tuple[str, str, str]:
    project = cfg.get("project", {})
    project_id = os.getenv("GCP_PROJECT_ID", project.get("project_id", ""))
    location = os.getenv("BIGQUERY_LOCATION", project.get("location", "US"))
    dataset = os.getenv("BIGQUERY_DATASET", project.get("dataset", "seller_copilot_prod"))
    if not project_id:
        raise RuntimeError("GCP project id missing. Set GCP_PROJECT_ID or config.project.project_id.")
    return project_id, location, dataset


def get_split_settings(cfg: dict[str, Any]) -> tuple[float, float, float, int]:
    p = cfg.get("pipeline", {})
    split = p.get("split", {})
    train = float(split.get("train", 0.8))
    val = float(split.get("val", 0.1))
    test = float(split.get("test", 0.1))
    seed = int(p.get("random_seed", 42))
    if abs((train + val + test) - 1.0) > 1e-8:
        raise RuntimeError("Train/val/test split ratios must sum to 1.0")
    return train, val, test, seed

