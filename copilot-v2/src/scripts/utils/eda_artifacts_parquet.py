from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

TARGET_SUBDIRS = ("data_snapshots", "features", "synthetic")


def _detect_default_artifacts_root() -> Path:
    # File is under copilot-v2/src/copilot_v2/scripts/, so parents[3] is copilot-v2/.
    return Path(__file__).resolve().parents[3] / "artifacts"


def _read_one(path: Path, sample_rows: int) -> pd.DataFrame:
    if sample_rows and sample_rows > 0:
        return pd.read_parquet(path).head(sample_rows)
    return pd.read_parquet(path)


def _col_profile(series: pd.Series, top_n: int) -> dict[str, Any]:
    null_count = int(series.isna().sum())
    unique_count = int(series.nunique(dropna=True))
    top_values = series.value_counts(dropna=True).head(top_n)
    top_map = {str(k): int(v) for k, v in top_values.items()}
    return {
        "dtype": str(series.dtype),
        "null_count": null_count,
        "null_ratio": round(float(null_count / max(len(series), 1)), 4),
        "unique_non_null": unique_count,
        "top_values": top_map,
    }


def _json_safe_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return _json_safe_value(value.item())
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def _collect_target_parquets(artifacts_root: Path) -> list[Path]:
    files: list[Path] = []
    for subdir in TARGET_SUBDIRS:
        base = artifacts_root / subdir
        if base.exists():
            files.extend(base.glob("**/*.parquet"))
    return sorted(set(files))


def inspect_parquet(path: Path, sample_rows: int, top_n: int, head_rows: int) -> dict[str, Any]:
    df = _read_one(path, sample_rows=sample_rows)
    preview_df = df.head(max(int(head_rows), 0)).copy()
    preview_df = preview_df.astype(object).where(pd.notna(preview_df), None)
    head_records = [
        {k: _json_safe_value(v) for k, v in row.items()}
        for row in preview_df.to_dict(orient="records")
    ]
    return {
        "file": str(path),
        "rows_loaded": int(len(df)),
        "columns": list(df.columns),
        "column_count": int(len(df.columns)),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "columns_profile": {col: _col_profile(df[col], top_n=top_n) for col in df.columns},
        "head": head_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick EDA for parquet files under artifacts/")
    parser.add_argument("--artifacts-root", type=Path, default=_detect_default_artifacts_root())
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=5000,
        help="Rows loaded per file (0 means full file)",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Top-N values shown per column")
    parser.add_argument("--head-rows", type=int, default=5, help="How many rows of head() to include")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for JSON report",
    )
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_root)
    if not artifacts_root.exists():
        raise FileNotFoundError(f"Artifacts root not found: {artifacts_root}")

    parquet_files = _collect_target_parquets(artifacts_root)
    if not parquet_files:
        print(
            json.dumps(
                {
                    "ok": False,
                    "reason": "No parquet files found in target subdirs",
                    "artifacts_root": str(artifacts_root),
                    "target_subdirs": list(TARGET_SUBDIRS),
                },
                indent=2,
            )
        )
        return

    report = {
        "ok": True,
        "artifacts_root": str(artifacts_root),
        "target_subdirs": list(TARGET_SUBDIRS),
        "n_parquet_files": int(len(parquet_files)),
        "sample_rows": int(args.sample_rows),
        "head_rows": int(args.head_rows),
        "files": [
            inspect_parquet(
                path,
                sample_rows=int(args.sample_rows),
                top_n=int(args.top_n),
                head_rows=int(args.head_rows),
            )
            for path in parquet_files
        ],
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"ok": True, "report_path": str(args.output_json), "n_files": len(parquet_files)}, indent=2))
        return

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
