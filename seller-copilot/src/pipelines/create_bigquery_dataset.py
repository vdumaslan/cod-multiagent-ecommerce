from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bq_utils import ensure_dataset, ensure_pipeline_runs_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--location", default="US")
    args = parser.parse_args()

    ensure_dataset(args.project_id, args.dataset, args.location)
    ensure_pipeline_runs_table(args.project_id, args.dataset)
    print(f"Dataset and admin tables ready: {args.project_id}.{args.dataset}")


if __name__ == "__main__":
    main()

