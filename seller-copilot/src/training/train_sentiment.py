from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--text-col", default="review_text")
    parser.add_argument("--label-col", default="rating")
    parser.add_argument(
        "--model-id", default="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    parser.add_argument("--output-dir", default="seller-copilot/artifacts/sentiment")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input_parquet)
    df = df.dropna(subset=[args.text_col, args.label_col]).copy()
    if df.empty:
        raise RuntimeError("No rows available after dropna.")

    # Simple 3-way target from rating for baseline evaluation.
    def to_label(r: float) -> int:
        if r >= 4:
            return 2
        if r == 3:
            return 1
        return 0

    df["y"] = df[args.label_col].astype(float).map(to_label)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["y"])

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id)
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer)

    preds = []
    for text in test_df[args.text_col].astype(str).tolist():
        p = clf(text[:512])[0]["label"].lower()
        if "negative" in p:
            preds.append(0)
        elif "neutral" in p:
            preds.append(1)
        else:
            preds.append(2)

    report = classification_report(test_df["y"], preds, output_dict=True)
    (out_dir / "classification_report.json").write_text(json.dumps(report, indent=2))
    print(f"Saved: {out_dir / 'classification_report.json'}")


if __name__ == "__main__":
    main()


