"""A/B testing support: variant assignment and event logging."""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

VARIANTS = ("A", "B")

_log_path: Path | None = None


def _get_log_path() -> Path:
    global _log_path
    if _log_path is None:
        root = os.environ.get("COPILOT_AB_LOG_DIR", "runs/ab_events")
        _log_path = Path(root)
        _log_path.mkdir(parents=True, exist_ok=True)
    return _log_path / "events.jsonl"


def assign_variant(owner_id: str) -> str:
    """Deterministically assign owner_id to 'A' or 'B'. Same owner always gets same variant."""
    digest = int(hashlib.md5(owner_id.encode()).hexdigest(), 16)
    return VARIANTS[digest % 2]


def save_version_a_run(
    *,
    owner_id: str,
    variant: str,
    goal: str,
    retrieval_candidates: list[dict],
    decisions: dict,
    time_to_decision_s: int | None,
    confidence_rating: int | None,
    runs_root: Path,
) -> str:
    """Save a Version A run to the runs directory. Returns the run_id."""
    ts = datetime.now(timezone.utc)
    owner_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", owner_id)[:16]
    run_id = f"{ts.strftime('%Y%m%d_%H%M%S')}_{owner_slug}_vA"
    run_dir = runs_root / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "ab_mode": "A",
        "variant": variant,
        "owner_id": owner_id,
        "goal": goal,
        "ts": ts.isoformat(),
    }
    (run_dir / "0_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "1_retrieval.json").write_text(json.dumps(retrieval_candidates, indent=2), encoding="utf-8")
    final = {
        "ab_mode": "A",
        "decisions": decisions,
        "time_to_decision_s": time_to_decision_s,
        "confidence_rating": confidence_rating,
    }
    (run_dir / "9_final.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    return run_id


def log_event(
    *,
    owner_id: str,
    variant: str,
    run_id: str,
    event: str,
    metadata: dict | None = None,
) -> None:
    """Append one event line to the JSONL log."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "owner_id": owner_id,
        "variant": variant,
        "run_id": run_id,
        "event": event,
        **(metadata or {}),
    }
    try:
        with _get_log_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass
