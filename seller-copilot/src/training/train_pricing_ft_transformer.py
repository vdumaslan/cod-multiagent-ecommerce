from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--target-col", default="price")
    parser.add_argument("--output-dir", default="seller-copilot/artifacts/pricing")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input_parquet)
    if args.target_col not in df.columns:
        raise RuntimeError(f"Target column '{args.target_col}' not found in input parquet.")

    # Placeholder baseline scaffold for FT-Transformer training integration.
    # Keep output schema stable so downstream evaluation/reporting is ready.
    y = pd.to_numeric(df[args.target_col], errors="coerce")
    y = y.dropna()
    if len(y) < 10:
        raise RuntimeError("Insufficient rows for pricing training scaffold.")

    train_y, test_y = train_test_split(y, test_size=0.2, random_state=42)
    # Naive baseline prediction to validate pipeline wiring.
    pred = [float(train_y.mean())] * len(test_y)

    mae = mean_absolute_error(test_y, pred)
    rmse = mean_squared_error(test_y, pred, squared=False)

    summary = {
        "model_id": "FT-Transformer",
        "status": "scaffold_ready",
        "notes": "Replace naive baseline with FT-Transformer training loop.",
        "metrics": {"mae": float(mae), "rmse": float(rmse)},
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved: {out_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()


