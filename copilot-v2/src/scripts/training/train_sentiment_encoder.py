from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rating_to_label(r: float) -> str | None:
    # Proxy labels used throughout the project.
    if r <= 2:
        return "neg"
    if r == 3:
        return "neu"
    if r >= 4:
        return "pos"
    return None


@dataclass(frozen=True)
class SentimentTrialConfig:
    snapshot_id: str
    artifacts_root: str
    model_id: str
    base_model_name: str
    experiment_slug: str
    run_tag: str
    max_train_rows: int
    max_val_rows: int
    eval_slice_10k_ids_path: str | None
    seed: int
    lr: float
    weight_decay: float
    warmup_ratio: float
    per_device_batch_size: int
    num_epochs: int
    max_length: int
    label_smoothing: float
    balanced_batches: bool
    device: str


def _require_transformers() -> Any:
    try:
        import torch  # noqa: F401
        from transformers import (  # type: ignore
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing sentiment training deps. Install transformers + torch (and accelerate). "
            "This script is intended to be run inside `.venv-copilot-v2`."
        ) from e
    return {
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorWithPadding": DataCollatorWithPadding,
        "EarlyStoppingCallback": EarlyStoppingCallback,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def _require_datasets() -> Any:
    try:
        from datasets import Dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing `datasets`. Install `datasets` in your venv.") from e
    return {"Dataset": Dataset}


def _balanced_downsample(df: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    # Keep class counts balanced by downsampling to the smallest class.
    g = df.groupby("label", group_keys=False)
    min_n = int(g.size().min()) if len(df) else 0
    if min_n <= 0:
        return df
    return g.apply(lambda x: x.sample(min_n, random_state=seed)).reset_index(drop=True)


def _compute_metrics_builder(labels_order: list[str]):
    def compute_metrics(eval_pred):
        try:
            from sklearn.metrics import (  # type: ignore
                accuracy_score,
                balanced_accuracy_score,
                f1_score,
                log_loss,
                matthews_corrcoef,
                confusion_matrix,
                roc_auc_score,
            )
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Missing sklearn; install scikit-learn.") from e

        logits, y_true = eval_pred
        logits = np.asarray(logits)
        y_true = np.asarray(y_true)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
        y_pred = probs.argmax(axis=1)

        macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
        acc = float(accuracy_score(y_true, y_pred))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        mcc = float(matthews_corrcoef(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels_order)))).tolist()
        ll = float(log_loss(y_true, probs, labels=list(range(len(labels_order)))))
        try:
            auc = float(roc_auc_score(y_true, probs, multi_class="ovr"))
        except Exception:
            auc = None

        return {
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "balanced_accuracy": bal_acc,
            "mcc": mcc,
            "accuracy": acc,
            "confusion_matrix": cm,
            "macro_roc_auc_ovr": auc,
            "log_loss": ll,
        }

    return compute_metrics


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def _metrics_filename(model_id: str, experiment_slug: str, split_name: str, run_tag: str, *, long: bool) -> str:
    # Match existing naming in artifacts:
    # - short: metrics_<model_id>_<experiment_slug>_<split>.json  (or metrics_<model_id>_<experiment_slug>_val.json)
    # - long : metrics_<model_id>_<experiment_slug>_<split>_<run_tag>.json
    if long:
        return f"metrics_{model_id}_{experiment_slug}_{split_name}_{run_tag}.json"
    return f"metrics_{model_id}_{experiment_slug}_{split_name}.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-id", required=True)
    p.add_argument("--artifacts-root", default="copilot-v2/artifacts")
    p.add_argument("--base-model-name", default="distilroberta-base")
    p.add_argument("--model-id", default="distilroberta-base")
    p.add_argument("--experiment-slug", required=True, help="E.g. phaseE_200k_wd005_wu010_len384 or final_500k_slice_winner")
    p.add_argument("--run-tag", default=None, help="Defaults to experiment-slug")
    p.add_argument("--max-train-rows", type=int, default=200000)
    p.add_argument("--max-val-rows", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--balanced-batches", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--slice-10k-ids",
        default=None,
        help="Optional path to sentiment_eval_slice_10k_ids.json; if omitted, will try artifacts default.",
    )
    args = p.parse_args()

    artifacts_root = Path(args.artifacts_root)
    snapshot_id = str(args.snapshot_id)
    run_tag = str(args.run_tag or args.experiment_slug)

    # Default slice list path (matches docs convention)
    slice_ids_path = args.slice_10k_ids
    if not slice_ids_path:
        cand = artifacts_root / "evals" / snapshot_id / "sentiment" / "sentiment_eval_slice_10k_ids.json"
        if cand.exists():
            slice_ids_path = str(cand)

    cfg = SentimentTrialConfig(
        snapshot_id=snapshot_id,
        artifacts_root=str(artifacts_root),
        model_id=str(args.model_id),
        base_model_name=str(args.base_model_name),
        experiment_slug=str(args.experiment_slug),
        run_tag=run_tag,
        max_train_rows=int(args.max_train_rows),
        max_val_rows=int(args.max_val_rows),
        eval_slice_10k_ids_path=slice_ids_path,
        seed=int(args.seed),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        warmup_ratio=float(args.warmup_ratio),
        per_device_batch_size=int(args.batch_size),
        num_epochs=int(args.epochs),
        max_length=int(args.max_length),
        label_smoothing=float(args.label_smoothing),
        balanced_batches=bool(args.balanced_batches),
        device=str(args.device),
    )

    # --- Load data ---
    reviews_path = artifacts_root / "data_snapshots" / snapshot_id / "reviews.parquet"
    splits_path = artifacts_root / "splits" / snapshot_id / "reviews_split.parquet"
    if not reviews_path.exists() or not splits_path.exists():
        raise FileNotFoundError(f"Missing reviews/splits. Expected {reviews_path} and {splits_path}.")

    rev = pd.read_parquet(reviews_path)
    sp = pd.read_parquet(splits_path)
    rev = rev.merge(sp, on=["review_id", "product_id"], how="inner")
    rev["label"] = rev["rating"].map(_rating_to_label)
    rev = rev.dropna(subset=["label"])
    rev = rev[rev["review_text"].astype(str).str.len() > 10].copy()

    labels_order = ["neg", "neu", "pos"]
    label_to_id = {lbl: i for i, lbl in enumerate(labels_order)}
    rev["label_id"] = rev["label"].map(label_to_id).astype(int)

    train_df = rev[rev["split"] == "train"].copy()
    val_df = rev[rev["split"] == "val"].copy()

    if cfg.balanced_batches:
        train_df = _balanced_downsample(train_df, seed=cfg.seed)

    if cfg.max_train_rows > 0 and len(train_df) > cfg.max_train_rows:
        train_df = train_df.sample(cfg.max_train_rows, random_state=cfg.seed).reset_index(drop=True)
    if cfg.max_val_rows > 0 and len(val_df) > cfg.max_val_rows:
        val_df = val_df.sample(cfg.max_val_rows, random_state=cfg.seed).reset_index(drop=True)

    # --- Prepare HF datasets ---
    tfm = _require_transformers()
    dsets = _require_datasets()
    AutoTokenizer = tfm["AutoTokenizer"]
    AutoModelForSequenceClassification = tfm["AutoModelForSequenceClassification"]
    DataCollatorWithPadding = tfm["DataCollatorWithPadding"]
    Trainer = tfm["Trainer"]
    TrainingArguments = tfm["TrainingArguments"]
    EarlyStoppingCallback = tfm["EarlyStoppingCallback"]
    Dataset = dsets["Dataset"]

    tok = AutoTokenizer.from_pretrained(cfg.base_model_name)

    def _tokenize(batch):
        return tok(
            batch["review_text"],
            truncation=True,
            max_length=cfg.max_length,
        )

    train_ds = Dataset.from_pandas(train_df[["review_text", "label_id"]].rename(columns={"label_id": "labels"}))
    val_ds = Dataset.from_pandas(val_df[["review_text", "label_id"]].rename(columns={"label_id": "labels"}))
    train_ds = train_ds.map(_tokenize, batched=True, remove_columns=["review_text"])
    val_ds = val_ds.map(_tokenize, batched=True, remove_columns=["review_text"])

    data_collator = DataCollatorWithPadding(tokenizer=tok)

    # Output layout (matches existing artifacts)
    model_out = artifacts_root / "models" / snapshot_id / "sentiment" / f"{cfg.model_id}_{cfg.experiment_slug}"
    eval_root = artifacts_root / "evals" / snapshot_id / "sentiment"
    trial_out = eval_root / "trial_runs" / cfg.run_tag

    # Device selection: leave accelerate/Trainer to decide for "auto".
    # If user forces cpu, set env for transformers/torch.
    if cfg.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model = AutoModelForSequenceClassification.from_pretrained(cfg.base_model_name, num_labels=len(labels_order))

    args_train = TrainingArguments(
        output_dir=str(model_out),
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        num_train_epochs=cfg.num_epochs,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        seed=cfg.seed,
        report_to=[],
        label_smoothing_factor=float(cfg.label_smoothing) if cfg.label_smoothing else 0.0,
    )

    compute_metrics = _compute_metrics_builder(labels_order)

    t0 = time.time()
    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    train_elapsed_s = time.time() - t0

    # --- Evaluate on val ---
    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    # trainer.evaluate returns keys like eval_macro_f1; normalize.
    def _strip_prefix(d: dict[str, Any], prefix: str) -> dict[str, Any]:
        out = {}
        for k, v in d.items():
            kk = k[len(prefix) :] if k.startswith(prefix) else k
            out[kk] = v
        return out

    eval_metrics = _strip_prefix({k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in val_metrics.items()}, "eval_")

    # Attach detailed fields to match existing artifact schema (schema_version=1).
    best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
    best_metric = getattr(trainer.state, "best_metric", None)
    global_step = int(getattr(trainer.state, "global_step", 0) or 0)

    payload_val: dict[str, Any] = {
        "schema_version": 1,
        "approach": "encoder_finetune",
        "model_id": cfg.model_id,
        "eval_name": "val_subsample",
        "snapshot_id": cfg.snapshot_id,
        "trial_id": cfg.run_tag,
        "run_tag": cfg.run_tag,
        "experiment_slug_arg": f"{cfg.model_id}_{cfg.experiment_slug}",
        "metrics_slug": f"{cfg.model_id}_{cfg.experiment_slug}",
        "skip_canonical_metrics": True,
        "n_evaluated": int(len(val_df)),
        "train_rows_used": int(len(train_df)),
        "max_train_rows_arg": cfg.max_train_rows,
        "max_val_rows_arg": cfg.max_val_rows,
        "learning_rate": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "per_device_batch_size": cfg.per_device_batch_size,
        "seed": cfg.seed,
        "final_val_full": False,
        "num_epochs_requested": cfg.num_epochs,
        "early_stopping_patience": 2,
        "trainer_best_metric": float(best_metric) if best_metric is not None and not math.isnan(float(best_metric)) else None,
        "trainer_best_model_checkpoint": str(best_ckpt) if best_ckpt else None,
        "trainer_global_step": global_step,
        "load_best_model_at_end": True,
        "balanced_batches": cfg.balanced_batches,
        "warmup_ratio": cfg.warmup_ratio,
        "label_smoothing": cfg.label_smoothing,
        "written_at_utc": _utc_now_iso(),
        "labels_order": labels_order,
        "n_samples": int(len(val_df)),
        "train_elapsed_s": float(train_elapsed_s),
    }
    payload_val.update(eval_metrics)

    _write_json(trial_out / "metrics_val.json", payload_val)

    # --- Evaluate on slice_10k if possible ---
    payload_slice = None
    if cfg.eval_slice_10k_ids_path and Path(cfg.eval_slice_10k_ids_path).exists():
        ids = json.loads(Path(cfg.eval_slice_10k_ids_path).read_text(encoding="utf-8"))
        if isinstance(ids, dict) and "review_id" in ids:
            ids = ids["review_id"]
        ids_set = set([str(x) for x in ids]) if isinstance(ids, list) else set()
        slice_df = rev[rev["review_id"].astype(str).isin(ids_set)].copy()
        if not slice_df.empty:
            slice_ds = Dataset.from_pandas(slice_df[["review_text", "label_id"]].rename(columns={"label_id": "labels"}))
            slice_ds = slice_ds.map(_tokenize, batched=True, remove_columns=["review_text"])
            slice_metrics = trainer.evaluate(eval_dataset=slice_ds)
            slice_metrics_n = _strip_prefix(
                {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in slice_metrics.items()},
                "eval_",
            )
            payload_slice = dict(payload_val)
            payload_slice["eval_name"] = "slice_10k"
            payload_slice["n_evaluated"] = int(len(slice_df))
            payload_slice["n_samples"] = int(len(slice_df))
            # Replace metric fields
            for k in list(payload_slice.keys()):
                if k in slice_metrics_n:
                    payload_slice[k] = slice_metrics_n[k]
            payload_slice.update(slice_metrics_n)
            _write_json(trial_out / "metrics_slice_10k.json", payload_slice)

    # --- Write canonical-style files under eval_root ---
    # These are used by MODEL_TRAINING_AND_RESULTS.md references.
    _write_json(eval_root / _metrics_filename(cfg.model_id, cfg.experiment_slug, "val", cfg.run_tag, long=False), payload_val)
    _write_json(eval_root / _metrics_filename(cfg.model_id, cfg.experiment_slug, "val", cfg.run_tag, long=True), payload_val)
    if payload_slice is not None:
        _write_json(eval_root / _metrics_filename(cfg.model_id, cfg.experiment_slug, "slice_10k", cfg.run_tag, long=False), payload_slice)
        _write_json(eval_root / _metrics_filename(cfg.model_id, cfg.experiment_slug, "slice_10k", cfg.run_tag, long=True), payload_slice)

    # Also persist the config for audit/debug.
    _write_json(trial_out / "trial_config.json", asdict(cfg))

    print(json.dumps({"ok": True, "model_out": str(model_out), "trial_out": str(trial_out)}, indent=2))


if __name__ == "__main__":
    main()

