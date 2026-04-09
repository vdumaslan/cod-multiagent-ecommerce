from __future__ import annotations

import argparse
from pathlib import Path

from copilot_v2.runtime.server import run_server


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-id", type=str, required=True)
    ap.add_argument("--artifacts-root", type=Path, default=Path("copilot-v2/artifacts"))
    ap.add_argument("--registry-path", type=Path, default=Path("copilot-v2/artifacts/registry/registry.json"))
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--grounding-cache-dir", type=Path, default=None)
    ap.add_argument("--demo-allowlist-json", type=Path, default=None)
    args = ap.parse_args()
    run_server(
        snapshot_id=args.snapshot_id,
        artifacts_root=args.artifacts_root,
        registry_path=args.registry_path,
        host=args.host,
        port=int(args.port),
        grounding_cache_dir=args.grounding_cache_dir,
        demo_allowlist_json=args.demo_allowlist_json,
    )


if __name__ == "__main__":
    main()

