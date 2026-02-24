from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def _label_from_model_output(label: str) -> int:
    t = label.lower()
    if "negative" in t:
        return 0
    if "neutral" in t:
        return 1
    return 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-parquet", default="seller-copilot/artifacts/data/sentiment_test.parquet")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--model-id", default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    parser.add_argument("--max-eval-rows", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default="seller-copilot/artifacts/sentiment")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_parquet(args.test_parquet).dropna(subset=[args.text_col, args.label_col]).copy()
    if test_df.empty:
        raise RuntimeError("No evaluation rows found in sentiment_test parquet.")
    if args.max_eval_rows and len(test_df) > args.max_eval_rows:
        test_df = test_df.sample(args.max_eval_rows, random_state=42).reset_index(drop=True)

    device = args.device
    device_id = -1
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    if device == "cuda":
        device_id = 0

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id)
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=256,
        batch_size=args.batch_size,
        device=device_id,
    )

    preds = clf(test_df[args.text_col].astype(str).tolist())
    y_pred = [_label_from_model_output(p["label"]) for p in preds]
    y_true = test_df[args.label_col].astype(int).tolist()

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    payload = {
        "model_id": args.model_id,
        "device": device,
        "num_eval_rows": int(len(test_df)),
        "metrics": report,
        "confusion_matrix_labels": [0, 1, 2],
        "confusion_matrix": cm,
    }
    (out_dir / "classification_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {out_dir / 'classification_report.json'}")


if __name__ == "__main__":
    main()
