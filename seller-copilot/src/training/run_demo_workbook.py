from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-dir", default="seller-copilot/artifacts")
    parser.add_argument("--sentiment-max-rows", type=int, default=100000)
    parser.add_argument("--ranking-max-rows", type=int, default=350000)
    parser.add_argument("--pricing-max-rows", type=int, default=150000)
    parser.add_argument("--reranker-epochs", type=int, default=0)
    parser.add_argument("--pricing-epochs", type=int, default=8)
    parser.add_argument("--embedding-max-rows", type=int, default=50000)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    prep = root / "prepare_datasets.py"
    sentiment = root / "train_sentiment.py"
    reranker = root / "train_reranker.py"
    pricing = root / "train_pricing_ft_transformer.py"
    embed = root / "build_embeddings.py"
    analytics = root / "analyze_data.py"
    evaluate = root / "evaluate.py"

    py = sys.executable
    data_dir = Path(args.output_dir) / "data"
    retrieval_dir = Path(args.output_dir) / "retrieval"

    prep_cmd = [
        py,
        str(prep),
        "--output-dir",
        str(data_dir),
        "--sentiment-max-rows",
        str(args.sentiment_max_rows),
        "--ranking-max-rows",
        str(args.ranking_max_rows),
        "--pricing-max-rows",
        str(args.pricing_max_rows),
    ]
    if args.project_id:
        prep_cmd += ["--project-id", args.project_id]
    if args.dataset:
        prep_cmd += ["--dataset", args.dataset]
    _run(prep_cmd)
    analytics_cmd = [py, str(analytics), "--output", str(Path(args.output_dir) / "data_analytics_summary.json")]
    if args.project_id:
        analytics_cmd += ["--project-id", args.project_id]
    if args.dataset:
        analytics_cmd += ["--dataset", args.dataset]
    _run(analytics_cmd)

    _run([py, str(sentiment), "--test-parquet", str(data_dir / "sentiment_test.parquet")])
    _run(
        [
            py,
            str(reranker),
            "--train-parquet",
            str(data_dir / "ranking_train.parquet"),
            "--test-parquet",
            str(data_dir / "ranking_test.parquet"),
            "--epochs",
            str(args.reranker_epochs),
        ]
    )
    _run(
        [
            py,
            str(pricing),
            "--train-parquet",
            str(data_dir / "pricing_train.parquet"),
            "--test-parquet",
            str(data_dir / "pricing_test.parquet"),
            "--epochs",
            str(args.pricing_epochs),
        ]
    )
    _run(
        [
            py,
            str(embed),
            "--input-parquet",
            str(data_dir / "retrieval_corpus.parquet"),
            "--max-rows",
            str(args.embedding_max_rows),
            "--output-dir",
            str(retrieval_dir),
        ]
    )
    _run([py, str(evaluate), "--ranking-test-parquet", str(data_dir / "ranking_test.parquet")])

    print("Demo/workbook artifact generation completed.")


if __name__ == "__main__":
    main()
