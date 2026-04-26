from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _rewrite_one(meta_path: Path) -> bool:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    paths = meta.get("paths")
    if not isinstance(paths, dict):
        return False

    base_dir = meta_path.parent.resolve()
    changed = False
    for k in ("corpus_parquet", "faiss_index", "doc_embeddings_npy"):
        v = paths.get(k)
        if not isinstance(v, str) or not v:
            continue
        p = Path(v)
        if not p.is_absolute():
            continue
        try:
            rel = Path(v).resolve().relative_to(base_dir)
        except Exception:
            # If the absolute path isn't under base_dir, do not rewrite (would break).
            continue
        paths[k] = str(rel)
        changed = True

    if changed:
        meta["paths"] = paths
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indexes-root", type=Path, required=True, help="Directory containing index_meta.json files (searched recursively).")
    args = ap.parse_args()

    root = Path(args.indexes_root)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    metas = sorted(root.rglob("index_meta.json"))
    if not metas:
        raise SystemExit(f"No index_meta.json found under: {root}")

    n_changed = 0
    for p in metas:
        if _rewrite_one(p):
            n_changed += 1
    print(json.dumps({"ok": True, "metas_found": len(metas), "metas_rewritten": n_changed}, indent=2))


if __name__ == "__main__":
    main()

