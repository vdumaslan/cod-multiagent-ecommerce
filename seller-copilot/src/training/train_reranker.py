from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--model-id", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--output-dir", default="seller-copilot/artifacts/reranker")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input_parquet)
    summary = {
        "model_id": args.model_id,
        "num_rows": int(len(df)),
        "status": "placeholder_training_script",
        "next_step": "Implement pairwise reranker fine-tuning if compute budget allows.",
    }

    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved: {out_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()


