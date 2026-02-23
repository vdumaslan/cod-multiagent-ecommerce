from __future__ import annotations

import argparse

from google.cloud import bigquery


def ensure_dataset(project_id: str, dataset_id: str, location: str) -> None:
    client = bigquery.Client(project=project_id)
    full_id = f"{project_id}.{dataset_id}"
    dataset = bigquery.Dataset(full_id)
    dataset.location = location
    client.create_dataset(dataset, exists_ok=True)
    print(f"Dataset ready: {full_id} ({location})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--location", default="US")
    args = parser.parse_args()
    ensure_dataset(args.project_id, args.dataset, args.location)


if __name__ == "__main__":
    main()

