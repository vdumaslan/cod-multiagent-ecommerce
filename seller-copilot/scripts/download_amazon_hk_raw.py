"""
Download Amazon Reviews 2023 — Home & Kitchen raw JSONL files from Hugging Face
to local disk (streaming, like a normal browser download).

Total size ~43 GB. Run and leave it until both files complete.

Usage:
  python seller-copilot/scripts/download_amazon_hk_raw.py
  python seller-copilot/scripts/download_amazon_hk_raw.py --out seller-copilot/data/raw/amazon_reviews_2023
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main"
FILES = [
    "raw/review_categories/Home_and_Kitchen.jsonl",
    "raw/meta_categories/meta_Home_and_Kitchen.jsonl",
]
CHUNK = 1024 * 1024  # 1 MiB


def download_one(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"SKIP (already exists, {dest.stat().st_size} bytes): {dest}", flush=True)
        return

    tmp = dest.with_suffix(dest.suffix + ".partial")
    print(f"GET {url}", flush=True)
    print(f" -> {dest}", flush=True)

    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = r.headers.get("Content-Length")
        total_i = int(total) if total else None
        written = 0
        t0 = time.time()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                if total_i and written % (50 * CHUNK) < CHUNK:
                    pct = 100.0 * written / total_i
                    mb = written / 1e6
                    print(f"  ... {mb:.1f} MB ({pct:.1f}%)", flush=True)
        tmp.rename(dest)

    dt = time.time() - t0
    print(f"DONE {dest.name}: {written} bytes in {dt:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="seller-copilot/data/raw/amazon_reviews_2023",
        help="Directory to write JSONL files into",
    )
    args = parser.parse_args()
    out = Path(args.out).resolve()

    for rel in FILES:
        url = f"{BASE}/{rel}"
        name = rel.split("/")[-1]
        download_one(url, out / name)

    print("All downloads finished.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
