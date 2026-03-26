#!/usr/bin/env python3
"""
Sentiment comparison: VADER vs DistilRoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest).

Proxy labels from rating: neg (<=2), neu (==3), pos (>=4).

Writes metrics + confusion matrices under artifacts/evals/sentiment/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from _paths import DATA_AGENT, EVALS_DIR, SPLITS_DIR


def rating_to_label(r: float) -> str | None:
    if r <= 2:
        return "neg"
    if r == 3:
        return "neu"
    if r >= 4:
        return "pos"
    return None


def vader_to_label(compound: float) -> str:
    if compound <= -0.05:
        return "neg"
    if compound >= 0.05:
        return "pos"
    return "neu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=12000, help="Max labeled rows for RoBERTa fine-eval")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    agent = DATA_AGENT
    rev = pd.read_parquet(agent / "reviews.parquet")
    sp = pd.read_parquet(SPLITS_DIR / "reviews_split.parquet")
    rev = rev.merge(sp, on=["review_id", "product_id"], how="inner")

    rev["label"] = rev["rating"].map(rating_to_label)
    rev = rev.dropna(subset=["label"])
    rev = rev[rev["review_text"].astype(str).str.len() > 10]

    train_df = rev[rev["split"] == "train"]
    test_df = rev[rev["split"] == "test"].copy()
    if len(test_df) > 5000:
        test_df = test_df.sample(5000, random_state=args.seed)

    y_true = test_df["label"].tolist()
    labels_order = ["neg", "neu", "pos"]

    out = EVALS_DIR / "sentiment"
    out.mkdir(parents=True, exist_ok=True)

    # --- VADER ---
    sia = SentimentIntensityAnalyzer()
    vader_pred = [vader_to_label(sia.polarity_scores(t)["compound"]) for t in test_df["review_text"]]
    v_f1 = f1_score(y_true, vader_pred, average="macro", labels=labels_order, zero_division=0)
    v_acc = accuracy_score(y_true, vader_pred)
    cm_v = confusion_matrix(y_true, vader_pred, labels=labels_order)
    (out / "metrics_vader.json").write_text(
        json.dumps({"macro_f1": v_f1, "accuracy": v_acc, "labels": labels_order}, indent=2),
        encoding="utf-8",
    )
    _plot_cm(cm_v, labels_order, out / "confusion_vader.png", "VADER")

    # --- RoBERTa (subset train for optional calibration; here zero-shot on test) ---
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    clf = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tok,
        device=-1,
        truncation=True,
        max_length=256,
    )

    id2label = model.config.id2label

    def label_id_to_y(lbl: str, score_idx: int | None = None) -> str:
        if lbl.startswith("LABEL_") and score_idx is None:
            idx = int(lbl.split("_")[-1])
            s = str(id2label.get(idx, lbl)).lower()
        else:
            s = lbl.lower()
        if "negative" in s:
            return "neg"
        if "positive" in s:
            return "pos"
        return "neu"

    texts = test_df["review_text"].astype(str).tolist()
    ro_preds: list[str] = []
    bs = args.batch_size
    for i in range(0, len(texts), bs):
        chunk = texts[i : i + bs]
        res = clf(chunk)
        if isinstance(res, dict):
            res = [res]
        for r in res:
            lbl = r["label"]
            ro_preds.append(label_id_to_y(str(lbl)))

    r_f1 = f1_score(y_true, ro_preds, average="macro", labels=labels_order, zero_division=0)
    r_acc = accuracy_score(y_true, ro_preds)
    cm_r = confusion_matrix(y_true, ro_preds, labels=labels_order)
    (out / "metrics_roberta.json").write_text(
        json.dumps(
            {
                "macro_f1": r_f1,
                "accuracy": r_acc,
                "labels": labels_order,
                "model": model_name,
                "note": "Zero-shot pipeline on test split; labels mapped to neg/neu/pos",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot_cm(cm_r, labels_order, out / "confusion_roberta.png", "DistilRoBERTa")

    summary = {
        "vader": {"macro_f1": v_f1, "accuracy": v_acc},
        "distil_roberta": {"macro_f1": r_f1, "accuracy": r_acc},
        "test_rows": len(test_df),
        "label_definition": "neg: rating<=2, neu: rating==3, pos: rating>=4",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _plot_cm(cm: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True",
        xlabel="Predicted",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
