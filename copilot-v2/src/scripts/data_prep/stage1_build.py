from __future__ import annotations

"""
Stage 1 ingestion entrypoint for Copilot V2.

This is a thin wrapper around the proven Stage 1 pipeline in `seller-copilot/`.
It exists so `Docs/DATA_PROCESSING_REPORT.md` can reference a runnable module under
`copilot_v2.scripts.*`.

Outputs are written to:
  copilot-v2/artifacts/data_snapshots/<snapshot_id>/

Files produced (minimum):
  - products.parquet
  - reviews.parquet
  - product_signals.parquet
  - retrieval_corpus.parquet
  - manifest.json (copilot-v2 wrapper manifest)

Note: the upstream Stage 1 flow writes an `agent_dataset_manifest.json`. We keep
that file and also write a Copilot V2 `manifest.json` for consistency with the
rest of this repo.
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    # copilot-v2/src/copilot_v2/scripts/stage1_build.py -> repo root
    return Path(__file__).resolve().parents[4]


def _ensure_seller_copilot_on_path(repo_root: Path) -> None:
    seller_src = repo_root / "seller-copilot" / "src"
    if str(seller_src) not in sys.path:
        sys.path.insert(0, str(seller_src))


def _read_jsonl_head(path: Path, n: int = 2) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
            if len(out) >= n:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-id", required=True, help="Snapshot id folder name to write under artifacts/data_snapshots/")
    ap.add_argument(
        "--reviews-jsonl",
        type=Path,
        default=Path("copilot-v2/data/raw/amazon_reviews_2023/Home_and_Kitchen.jsonl"),
    )
    ap.add_argument(
        "--meta-jsonl",
        type=Path,
        default=Path("copilot-v2/data/raw/amazon_reviews_2023/meta_Home_and_Kitchen.jsonl"),
    )
    ap.add_argument(
        "--seller-config",
        type=Path,
        default=Path("seller-copilot/config/stage1_local_agent.yaml"),
        help="Seller-copilot Stage1 YAML config (we override local paths + output dir via env).",
    )
    ap.add_argument("--artifacts-root", type=Path, default=Path("copilot-v2/artifacts"))
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing snapshot folder if present.")
    args = ap.parse_args()

    repo_root = _repo_root()
    _ensure_seller_copilot_on_path(repo_root)

    reviews_jsonl = (repo_root / args.reviews_jsonl).resolve() if not args.reviews_jsonl.is_absolute() else args.reviews_jsonl
    meta_jsonl = (repo_root / args.meta_jsonl).resolve() if not args.meta_jsonl.is_absolute() else args.meta_jsonl
    seller_cfg = (repo_root / args.seller_config).resolve() if not args.seller_config.is_absolute() else args.seller_config
    artifacts_root = (repo_root / args.artifacts_root).resolve() if not args.artifacts_root.is_absolute() else args.artifacts_root

    out_dir = artifacts_root / "data_snapshots" / str(args.snapshot_id)
    if out_dir.exists():
        if args.overwrite:
            shutil.rmtree(out_dir)
        else:
            raise SystemExit(f"Output dir already exists: {out_dir}. Pass --overwrite to replace.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Wire Stage 1 using env overrides that `seller-copilot/src/data_acquisition/config.py` supports.
    os.environ["STAGE1_USE_LOCAL_FILES"] = "true"
    os.environ["STAGE1_LOCAL_REVIEWS_PATH"] = str(reviews_jsonl)
    os.environ["STAGE1_LOCAL_META_PATH"] = str(meta_jsonl)
    os.environ["STAGE1_AGENT_DATASET_DIR"] = str(out_dir)

    from data_acquisition.flows.stage1_amazon_hk_flow import run_stage1  # type: ignore

    summary = run_stage1(config_path=str(seller_cfg))

    # Normalize expected filenames to Copilot V2 conventions.
    # Seller-copilot writes: products.parquet, reviews.parquet, product_signals.parquet, retrieval_corpus.parquet, agent_dataset_manifest.json
    # We add: manifest.json (Copilot V2 wrapper)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": str(args.snapshot_id),
        "source": {
            "reviews_jsonl": str(reviews_jsonl),
            "meta_jsonl": str(meta_jsonl),
            "seller_config": str(seller_cfg),
            "notes": "Generated via copilot_v2.scripts.stage1_build (wrapper over seller-copilot Stage 1).",
        },
        "row_counts": summary.get("agent_row_counts") if isinstance(summary, dict) else None,
        "output_dir": str(out_dir),
        "raw_jsonl_head": {
            "reviews": _read_jsonl_head(reviews_jsonl, n=1),
            "meta": _read_jsonl_head(meta_jsonl, n=1),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({"ok": True, "out_dir": str(out_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

