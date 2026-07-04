from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime.orchestrator import _default_owner_for_product


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--response-json", type=Path, required=True, help="Saved orchestrator response JSON.")
    ap.add_argument("--owner-id", type=str, required=True)
    ap.add_argument("--n-owners", type=int, default=8)
    args = ap.parse_args()

    d = json.loads(args.response_json.read_text(encoding="utf-8"))
    assert d.get("ok") is True, f"Response not ok: {d.get('error')}"
    trace = d.get("trace") or {}
    assert trace.get("owner_id") == args.owner_id, f"trace.owner_id mismatch: {trace.get('owner_id')} != {args.owner_id}"

    acts = d.get("ranked_actions") or []
    assert acts, "No ranked_actions returned."

    bad = []
    for a in acts:
        pid = str(a.get("product_id") or "")
        if not pid:
            bad.append({"reason": "missing_product_id", "action": a})
            continue
        oid = _default_owner_for_product(pid, n_owners=int(args.n_owners))
        if oid != args.owner_id:
            bad.append({"reason": "wrong_owner", "product_id": pid, "expected": args.owner_id, "got": oid})

    assert not bad, f"Found actions not belonging to owner: {bad[:3]}"
    print(json.dumps({"ok": True, "owner_id": args.owner_id, "n_actions": len(acts)}, indent=2))


if __name__ == "__main__":
    main()

