from __future__ import annotations

import subprocess
from pathlib import Path

from prefect import flow, task


ROOT = Path(__file__).resolve().parents[2]
PIPELINES = ROOT / "src" / "pipelines"


@task
def ingest_raw(project_id: str, dataset: str) -> None:
    cmd = [
        "python",
        str(PIPELINES / "ingest_sources.py"),
        "--project-id",
        project_id,
        "--dataset",
        dataset,
    ]
    subprocess.run(cmd, check=True)


@task
def ensure_dataset(project_id: str, dataset: str, location: str) -> None:
    cmd = [
        "python",
        str(PIPELINES / "create_bigquery_dataset.py"),
        "--project-id",
        project_id,
        "--dataset",
        dataset,
        "--location",
        location,
    ]
    subprocess.run(cmd, check=True)


@task
def run_sql(script_path: Path, project_id: str, dataset: str) -> None:
    from google.cloud import bigquery

    sql = script_path.read_text(encoding="utf-8")
    sql = sql.replace("{PROJECT_ID}", project_id).replace("{DATASET}", dataset)

    client = bigquery.Client(project=project_id)
    for statement in [s.strip() for s in sql.split(";") if s.strip()]:
        client.query(statement).result()
    print(f"Executed SQL script: {script_path.name}")


@task
def write_quality_report(project_id: str, dataset: str) -> None:
    cmd = [
        "python",
        str(PIPELINES / "data_quality_report.py"),
        "--project-id",
        project_id,
        "--dataset",
    ]
    subprocess.run(cmd, check=True)


@flow(name="cod_fresh_start_pipeline")
def cod_pipeline(project_id: str, dataset: str, location: str = "US") -> None:
    ensure_dataset(project_id, dataset, location)
    ingest_raw(project_id, dataset)
    run_sql(PIPELINES / "transform_bigquery.sql", project_id, dataset)
    run_sql(PIPELINES / "build_features.sql", project_id, dataset)
    write_quality_report(project_id, dataset)


if __name__ == "__main__":
    # Update these two values to your own BigQuery project/dataset.
    cod_pipeline(project_id="your-gcp-project-id", dataset="cod_fresh_start", location="US")
