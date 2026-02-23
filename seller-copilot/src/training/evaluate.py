from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="seller-copilot/artifacts/metrics_summary.json")
    args = parser.parse_args()

    # Placeholder structure to standardize metrics reporting.
    metrics = {
        "retrieval": {"recall_at_10": None, "ndcg_at_10": None, "mrr": None},
        "ranking": {"ndcg_at_10": None, "pairwise_accuracy": None},
        "sentiment": {"macro_f1": None},
        "pricing": {"mae": None, "rmse": None, "value_score_calibration": None},
        "end_to_end": {"success_at_1": None, "latency_p50_ms": None, "latency_p95_ms": None},
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

