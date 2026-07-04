from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class GroundingCacheLoadResult:
    cache_dir: Path
    source: str  # local_dir | gs_uri | unknown
    loaded_pricing: bool
    loaded_sentiment: bool


def _stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    # gs://bucket/path/to/dir
    u = uri.strip()
    if not u.startswith("gs://"):
        raise ValueError(f"Not a gs:// uri: {uri}")
    rest = u[len("gs://") :]
    if "/" not in rest:
        return rest, ""
    bucket, key_prefix = rest.split("/", 1)
    return bucket, key_prefix.rstrip("/") + "/"


def _ensure_local_dir_for_uri(uri: str, *, local_root: Path | None = None) -> Path:
    root = local_root or Path(os.getenv("COPILOT_V2_CACHE_LOCAL_ROOT") or "/tmp/copilot_v2_cache")
    root.mkdir(parents=True, exist_ok=True)
    return (root / _stable_hash(uri)).resolve()


def _download_gcs_dir_via_gsutil(*, uri: str, dst_dir: Path) -> None:
    # Requires gsutil installed + auth configured (e.g., service account / ADC).
    dst_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["gsutil", "-m", "cp", "-r", f"{uri.rstrip('/')}/", str(dst_dir)]
    subprocess.check_call(cmd)


def _download_gcs_dir_via_python(*, uri: str, dst_dir: Path, filenames: list[str]) -> None:
    try:
        from google.cloud import storage  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "To load caches from gs:// URIs, install google-cloud-storage "
            "or ensure `gsutil` is available in the runtime image."
        ) from e

    bucket_name, prefix = _parse_gs_uri(uri)
    dst_dir.mkdir(parents=True, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for name in filenames:
        blob = bucket.blob(prefix + name)
        if not blob.exists(client=client):
            continue
        blob.download_to_filename(str((dst_dir / name).resolve()))


def resolve_grounding_cache_dir(*, grounding_cache_dir: Path | None, grounding_cache_uri: str | None) -> tuple[Path | None, str]:
    """
    Resolve where caches will be read from.

    Returns (local_cache_dir_or_none, source_label).
    """
    if grounding_cache_uri:
        uri = str(grounding_cache_uri).strip()
        if uri.startswith("gs://"):
            local_dir = _ensure_local_dir_for_uri(uri)
            # If already present, reuse.
            if not local_dir.exists() or not any(local_dir.iterdir()):
                try:
                    _download_gcs_dir_via_python(
                        uri=uri,
                        dst_dir=local_dir,
                        filenames=["pricing_cache.json", "sentiment_cache.json", "pricing_cache.parquet", "sentiment_cache.parquet"],
                    )
                except Exception:
                    # Fallback to gsutil if python client isn't available.
                    _download_gcs_dir_via_gsutil(uri=uri, dst_dir=local_dir)
            return local_dir, "gs_uri"

        # For now: treat as local directory path.
        p = Path(uri)
        return p, "local_dir"

    if grounding_cache_dir is not None:
        return Path(grounding_cache_dir), "local_dir"
    return None, "unknown"


def _load_pricing_cache(cache_dir: Path) -> dict[str, float] | None:
    pj = cache_dir / "pricing_cache.json"
    pp = cache_dir / "pricing_cache.parquet"
    if pj.is_file():
        raw = json.loads(pj.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return {str(k): float(v) for k, v in raw.items()}
        return None
    if pp.is_file():
        df = pd.read_parquet(pp)
        if "product_id" not in df.columns:
            return None
        value_col = "predicted_price_change_pct" if "predicted_price_change_pct" in df.columns else None
        if value_col is None:
            # allow "value" as generic
            value_col = "value" if "value" in df.columns else None
        if value_col is None:
            return None
        out: dict[str, float] = {}
        for pid, v in zip(df["product_id"].astype(str).tolist(), pd.to_numeric(df[value_col], errors="coerce").tolist(), strict=False):
            if v is None:
                continue
            try:
                out[str(pid)] = float(v)
            except Exception:
                continue
        return out
    return None


def _load_sentiment_cache(cache_dir: Path) -> dict[str, dict[str, float]] | None:
    sj = cache_dir / "sentiment_cache.json"
    sp = cache_dir / "sentiment_cache.parquet"
    if sj.is_file():
        raw = json.loads(sj.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            out: dict[str, dict[str, float]] = {}
            for pid, payload in raw.items():
                if isinstance(payload, dict):
                    out[str(pid)] = {str(k): float(v) for k, v in payload.items()}
            return out
        return None
    if sp.is_file():
        df = pd.read_parquet(sp)
        if "product_id" not in df.columns:
            return None
        # Two accepted shapes:
        # 1) Wide columns (p_pos, p_neu, p_neg, n_reviews, ...)
        # 2) payload_json column containing dict-like JSON
        if "payload_json" in df.columns:
            out: dict[str, dict[str, float]] = {}
            for pid, js in zip(df["product_id"].astype(str).tolist(), df["payload_json"].tolist(), strict=False):
                try:
                    payload = json.loads(js) if isinstance(js, str) else (js if isinstance(js, dict) else None)
                except Exception:
                    payload = None
                if isinstance(payload, dict):
                    out[str(pid)] = {str(k): float(v) for k, v in payload.items() if v is not None}
            return out

        wide_cols = [c for c in df.columns if c != "product_id"]
        out2: dict[str, dict[str, float]] = {}
        for _, row in df.iterrows():
            pid = str(row["product_id"])
            payload: dict[str, float] = {}
            for c in wide_cols:
                v = row[c]
                try:
                    payload[str(c)] = float(v)
                except Exception:
                    continue
            if payload:
                out2[pid] = payload
        return out2
    return None


def load_grounding_caches_into_ctx(*, ctx: Any, cache_dir: Path) -> GroundingCacheLoadResult:
    # Support two layouts:
    # 1) Flat dir containing pricing_cache.* and sentiment_cache.*
    # 2) Structured cache root:
    #    <cache_dir>/pricing/pricing_cache.*
    #    <cache_dir>/sentiment/sentiment_cache.*
    pricing = _load_pricing_cache(cache_dir)
    sentiment = _load_sentiment_cache(cache_dir)
    if pricing is None:
        pricing = _load_pricing_cache(cache_dir / "pricing")
    if sentiment is None:
        sentiment = _load_sentiment_cache(cache_dir / "sentiment")
    if pricing is not None:
        ctx.pricing_cache = pricing
    if sentiment is not None:
        ctx.sentiment_cache = sentiment
    return GroundingCacheLoadResult(
        cache_dir=cache_dir,
        source="local_dir",
        loaded_pricing=pricing is not None,
        loaded_sentiment=sentiment is not None,
    )

