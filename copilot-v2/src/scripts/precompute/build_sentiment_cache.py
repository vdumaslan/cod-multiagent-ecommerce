from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_registry(registry_path: Path) -> dict:
    return json.loads(registry_path.read_text(encoding="utf-8"))


def _require_transformers_and_torch() -> dict:
    try:
        import torch  # type: ignore
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing deps for distilroberta sentiment cache build. "
            "Install `torch` + `transformers` in the environment running this script."
        ) from e
    return {
        "torch": torch,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
    }


def _rating_bucket_cache(rev: pd.DataFrame) -> pd.DataFrame:
    rev = rev.copy()
    rev["rating"] = pd.to_numeric(rev["rating"], errors="coerce")
    rev = rev.dropna(subset=["product_id", "rating"]).copy()
    rev["product_id"] = rev["product_id"].astype(str)

    # Buckets match docs conventions: neg<=2, neu==3, pos>=4
    r = rev["rating"].to_numpy(dtype=float)
    label = np.where(r <= 2.0, "neg", np.where(r >= 4.0, "pos", "neu"))
    rev["label"] = label

    g = rev.groupby(["product_id", "label"], as_index=False).size()
    pivot = g.pivot(index="product_id", columns="label", values="size").fillna(0.0)
    pivot.columns = [str(c) for c in pivot.columns]
    for c in ["neg", "neu", "pos"]:
        if c not in pivot.columns:
            pivot[c] = 0.0

    pivot["n_reviews"] = pivot["neg"] + pivot["neu"] + pivot["pos"]
    denom = pivot["n_reviews"].replace(0.0, np.nan)
    pivot["p_neg"] = (pivot["neg"] / denom).fillna(0.0)
    pivot["p_neu"] = (pivot["neu"] / denom).fillna(0.0)
    pivot["p_pos"] = (pivot["pos"] / denom).fillna(0.0)

    out = pivot.reset_index()[["product_id", "n_reviews", "p_pos", "p_neu", "p_neg"]].copy()
    out["n_reviews"] = out["n_reviews"].astype(int)
    return out


def _distilroberta_cache_from_reviews_parquet(
    *,
    reviews_path: Path,
    model_dir: Path,
    device: str,
    batch_size: int,
    max_length: int,
    max_reviews: int,
    max_reviews_per_product: int,
) -> pd.DataFrame:
    # Stream review_text in batches (avoid loading 5.3M rows into RAM).
    deps = _require_transformers_and_torch()
    torch = deps["torch"]
    AutoModelForSequenceClassification = deps["AutoModelForSequenceClassification"]
    AutoTokenizer = deps["AutoTokenizer"]

    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing pyarrow; required for streaming parquet sentiment inference.") from e

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    if device == "cuda" and torch.cuda.is_available():
        mdl = mdl.to("cuda")
        dev = "cuda"
    else:
        mdl = mdl.to("cpu")
        dev = "cpu"
    mdl.eval()

    # Determine label mapping.
    id2label = getattr(mdl.config, "id2label", None) or {}
    # Normalize labels to {neg, neu, pos}
    def _norm(lbl: str) -> str:
        s = str(lbl).lower()
        if "neg" in s:
            return "neg"
        if "neu" in s:
            return "neu"
        if "pos" in s:
            return "pos"
        return s

    # Fallback order if config missing.
    label_order = [_norm(id2label.get(i, i)) for i in range(int(getattr(mdl.config, "num_labels", 3)))]
    if set(label_order) != {"neg", "neu", "pos"}:
        # Common training convention from our code: 3-way classification.
        label_order = ["neg", "neu", "pos"]

    counts: dict[str, dict[str, int]] = {}
    seen_per_pid: dict[str, int] = {}

    pf = pq.ParquetFile(str(reviews_path))
    seen = 0
    kept = 0
    for batch in pf.iter_batches(columns=["product_id", "review_text"], batch_size=int(batch_size) * 32):
        df = batch.to_pandas()
        if max_reviews > 0 and seen >= max_reviews:
            break
        if max_reviews > 0 and (seen + len(df)) > max_reviews:
            df = df.iloc[: max_reviews - seen].copy()

        df = df.dropna(subset=["product_id", "review_text"]).copy()
        if df.empty:
            seen += len(batch)
            continue
        pids_all = df["product_id"].astype(str).tolist()
        texts_all = df["review_text"].astype(str).tolist()

        # Apply per-product cap (keeps cache practical).
        if int(max_reviews_per_product) > 0:
            pids: list[str] = []
            texts: list[str] = []
            for pid, txt in zip(pids_all, texts_all, strict=False):
                c = int(seen_per_pid.get(pid, 0))
                if c >= int(max_reviews_per_product):
                    continue
                seen_per_pid[pid] = c + 1
                pids.append(pid)
                texts.append(txt)
            if not pids:
                seen += len(batch)
                continue
        else:
            pids = pids_all
            texts = texts_all

        # Mini-batch within the parquet batch.
        for i in range(0, len(texts), int(batch_size)):
            t_chunk = texts[i : i + int(batch_size)]
            p_chunk = pids[i : i + int(batch_size)]
            enc = tok(
                t_chunk,
                truncation=True,
                max_length=int(max_length),
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.no_grad():
                logits = mdl(**enc).logits
                pred = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(int).tolist()
            for pid, cls_id in zip(p_chunk, pred, strict=False):
                lbl = label_order[int(cls_id)] if int(cls_id) < len(label_order) else "neu"
                d = counts.get(pid)
                if d is None:
                    d = {"neg": 0, "neu": 0, "pos": 0}
                    counts[pid] = d
                if lbl not in d:
                    lbl = "neu"
                d[lbl] += 1
                kept += 1

        seen += len(batch)
        if kept and kept % 200_000 == 0:
            print(json.dumps({"progress": True, "reviews_seen": int(seen), "reviews_kept": int(kept), "unique_products": int(len(counts))}))

    rows = []
    for pid, d in counts.items():
        n = int(d.get("neg", 0) + d.get("neu", 0) + d.get("pos", 0))
        if n <= 0:
            continue
        rows.append(
            {
                "product_id": str(pid),
                "n_reviews": n,
                "p_pos": float(d.get("pos", 0) / n),
                "p_neu": float(d.get("neu", 0) / n),
                "p_neg": float(d.get("neg", 0) / n),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["n_reviews"] = out["n_reviews"].astype(int)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-id", required=True)
    p.add_argument("--artifacts-root", default="copilot-v2/artifacts")
    p.add_argument("--registry-path", default="copilot-v2/artifacts/registry/registry.json")
    p.add_argument(
        "--approach",
        default="ratings",
        choices=["ratings", "distilroberta"],
        help="ratings = fast rating-bucket cache; distilroberta = winner model inference on review_text.",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Default: <artifacts_root>/caches/<snapshot_id>/sentiment/",
    )
    p.add_argument("--write-json", action="store_true", help="Also write sentiment_cache.json mapping product_id->dict.")
    p.add_argument(
        "--reviews-path",
        default="",
        help="Override reviews.parquet path. Default: <artifacts_root>/data_snapshots/<snapshot_id>/reviews.parquet",
    )
    p.add_argument("--max-reviews", type=int, default=0, help="Optional cap for faster local rebuilds (0 = no cap).")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--batch-size", type=int, default=64, help="Only used for distilroberta approach.")
    p.add_argument("--max-length", type=int, default=256, help="Only used for distilroberta approach.")
    p.add_argument("--model-dir", default="", help="Override model dir for distilroberta approach (else from registry).")
    p.add_argument(
        "--max-reviews-per-product",
        type=int,
        default=24,
        help="Only used for distilroberta approach. 0 means no per-product cap (slow; scores all reviews).",
    )
    args = p.parse_args()

    artifacts_root = Path(args.artifacts_root)
    snapshot_id = str(args.snapshot_id)

    reviews_path = (
        Path(args.reviews_path)
        if str(args.reviews_path).strip()
        else (artifacts_root / "data_snapshots" / snapshot_id / "reviews.parquet")
    )
    if not reviews_path.is_file():
        raise FileNotFoundError(f"Missing {reviews_path}")

    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (artifacts_root / "caches" / snapshot_id / "sentiment")
    out_dir.mkdir(parents=True, exist_ok=True)

    approach = str(args.approach).strip().lower()
    if approach == "distilroberta":
        registry = _read_registry(Path(args.registry_path))
        model_dir = Path(args.model_dir) if str(args.model_dir).strip() else Path(str(registry["sentiment"]["model_dir"]))
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"Missing sentiment model_dir at {model_dir}. "
                "Download/unzip model artifacts first, or pass --model-dir."
            )
        device = str(args.device)
        if device == "auto":
            # Avoid importing torch here; resolve in the helper.
            device = "cuda"
        out = _distilroberta_cache_from_reviews_parquet(
            reviews_path=reviews_path,
            model_dir=model_dir,
            device=device,
            batch_size=int(args.batch_size),
            max_length=int(args.max_length),
            max_reviews=int(args.max_reviews),
            max_reviews_per_product=int(args.max_reviews_per_product),
        )
    else:
        use_cols = ["product_id", "rating"]
        rev = pd.read_parquet(reviews_path, columns=use_cols)
        if int(args.max_reviews) > 0 and len(rev) > int(args.max_reviews):
            rev = rev.sample(int(args.max_reviews), random_state=42).reset_index(drop=True)
        out = _rating_bucket_cache(rev)

    out_path = out_dir / "sentiment_cache.parquet"
    out.to_parquet(out_path, index=False)

    meta = {
        "schema_version": 1,
        "created_at_utc": _utc_now_iso(),
        "snapshot_id": snapshot_id,
        "source_reviews": str(reviews_path),
        "max_reviews": int(args.max_reviews),
        "rows": int(len(out)),
        "columns": list(out.columns),
        "approach": approach,
        "label_buckets": {"neg": "<=2", "neu": "==3", "pos": ">=4"},
    }
    (out_dir / "sentiment_cache_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if bool(args.write_json):
        mapping = {
            str(r["product_id"]): {
                "n_reviews": float(r["n_reviews"]),
                "p_pos": float(r["p_pos"]),
                "p_neu": float(r["p_neu"]),
                "p_neg": float(r["p_neg"]),
            }
            for _, r in out.iterrows()
        }
        (out_dir / "sentiment_cache.json").write_text(json.dumps(mapping), encoding="utf-8")

    print(json.dumps({"ok": True, "out_dir": str(out_dir), "parquet": str(out_path), "rows": int(len(out))}, indent=2))


if __name__ == "__main__":
    main()

