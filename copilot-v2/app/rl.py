"""Contextual bandit for tuning Version B policy knobs (RL-lite).

Goal: maximize acceptance / reduce abandonment and improve confidence.

This is intentionally simple and local-friendly:
- Uses a small discrete set of policy "arms".
- Selects via UCB on a scalar reward.
- Persists state as JSON under runs/rl/ by default.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RLArm:
    arm_id: str
    name: str
    config: dict[str, Any]


def default_arms() -> list[RLArm]:
    """Discrete policy profiles the bandit can choose from."""
    return [
        RLArm(
            arm_id="strict",
            name="Strict retrieval + rewrite",
            config={"retrieval_min_score": 0.80, "enable_query_rewrite": True, "enable_pricing_shrinkage": True},
        ),
        RLArm(
            arm_id="balanced",
            name="Balanced retrieval + rewrite",
            config={"retrieval_min_score": 0.75, "enable_query_rewrite": True, "enable_pricing_shrinkage": True},
        ),
        RLArm(
            arm_id="soft",
            name="Soft retrieval + rewrite",
            config={"retrieval_min_score": 0.70, "enable_query_rewrite": True, "enable_pricing_shrinkage": True},
        ),
        RLArm(
            arm_id="no_rewrite",
            name="Balanced retrieval + no rewrite",
            config={"retrieval_min_score": 0.75, "enable_query_rewrite": False, "enable_pricing_shrinkage": True},
        ),
        RLArm(
            arm_id="no_shrink",
            name="Balanced retrieval + rewrite + no shrinkage",
            config={"retrieval_min_score": 0.75, "enable_query_rewrite": True, "enable_pricing_shrinkage": False},
        ),
    ]


def _state_path() -> Path:
    root = os.environ.get("COPILOT_RL_STATE_DIR", "runs/rl").strip() or "runs/rl"
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    return p / "bandit_state.json"


def load_state(arms: list[RLArm]) -> dict[str, Any]:
    path = _state_path()
    if not path.exists():
        return {
            "schema_version": 1,
            "created_at_utc": _utc_now_iso(),
            "updated_at_utc": _utc_now_iso(),
            "total_updates": 0,
            "arms": {a.arm_id: {"n": 0, "mean_reward": 0.0} for a in arms},
            "runs": {},  # run_id -> {"arm_id":..., "partial": {...}}
        }
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("bad_state")
        obj.setdefault("arms", {})
        for a in arms:
            obj["arms"].setdefault(a.arm_id, {"n": 0, "mean_reward": 0.0})
        obj.setdefault("runs", {})
        return obj
    except Exception:
        # If corrupted, start fresh rather than crash the app.
        return {
            "schema_version": 1,
            "created_at_utc": _utc_now_iso(),
            "updated_at_utc": _utc_now_iso(),
            "total_updates": 0,
            "arms": {a.arm_id: {"n": 0, "mean_reward": 0.0} for a in arms},
            "runs": {},
        }


def save_state(state: dict[str, Any]) -> None:
    path = _state_path()
    state["updated_at_utc"] = _utc_now_iso()
    try:
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception:
        pass


def select_arm(*, arms: list[RLArm], state: dict[str, Any]) -> RLArm:
    """UCB selection on scalar reward."""
    arms_state: dict[str, Any] = state.get("arms") or {}
    total_n = sum(int((arms_state.get(a.arm_id) or {}).get("n", 0)) for a in arms)

    # Explore any untried arm first.
    for a in arms:
        n = int((arms_state.get(a.arm_id) or {}).get("n", 0))
        if n <= 0:
            return a

    # UCB
    c = float(os.environ.get("COPILOT_RL_UCB_C", "1.2") or 1.2)
    best = arms[0]
    best_score = -1e18
    for a in arms:
        st = arms_state.get(a.arm_id) or {}
        n = max(1, int(st.get("n", 0)))
        mean = float(st.get("mean_reward", 0.0))
        bonus = c * math.sqrt(max(1.0, math.log(max(2, total_n))) / float(n))
        score = mean + bonus
        if score > best_score:
            best_score = score
            best = a
    return best


def attach_run(state: dict[str, Any], *, run_id: str, arm_id: str) -> None:
    runs = state.setdefault("runs", {})
    runs[str(run_id)] = {"arm_id": str(arm_id), "partial": {}}


def compute_reward(*, decided: bool, abandoned: bool, confidence: int | None) -> float:
    """Reward emphasizing acceptance + confidence (as requested)."""
    r = 0.0
    r += 2.0 if decided else 0.0
    r -= 2.0 if abandoned else 0.0
    if confidence is not None:
        try:
            c = int(confidence)
        except Exception:
            c = 3
        c = max(1, min(5, c))
        r += 0.6 * float(c - 3)
    return float(r)


def update_from_event(
    state: dict[str, Any],
    *,
    run_id: str,
    event: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update bandit state from AB events. Returns info about any completed update."""
    meta = metadata or {}
    runs = state.setdefault("runs", {})
    rec = runs.get(str(run_id))
    if not rec:
        return {"ok": False, "updated": False, "reason": "unknown_run_id"}
    arm_id = str(rec.get("arm_id") or "")
    partial = rec.setdefault("partial", {})

    if event == "decision_made":
        partial["decided"] = True
    elif event == "abandoned":
        partial["abandoned"] = True
    elif event == "confidence_rated":
        rating = meta.get("rating", None)
        partial["confidence"] = rating

    decided = bool(partial.get("decided", False))
    abandoned = bool(partial.get("abandoned", False))
    confidence = partial.get("confidence", None)
    # Finalize when we have either (abandoned) OR (decided + confidence)
    done = abandoned or (decided and confidence is not None)
    if not done:
        return {"ok": True, "updated": False}

    reward = compute_reward(decided=decided, abandoned=abandoned, confidence=confidence if confidence is not None else None)
    arms_state = state.setdefault("arms", {})
    st = arms_state.setdefault(arm_id, {"n": 0, "mean_reward": 0.0})
    n0 = int(st.get("n", 0))
    m0 = float(st.get("mean_reward", 0.0))
    n1 = n0 + 1
    m1 = m0 + (reward - m0) / float(n1)
    st["n"] = n1
    st["mean_reward"] = float(m1)
    state["total_updates"] = int(state.get("total_updates", 0)) + 1

    # Remove run record after update to keep state small.
    try:
        runs.pop(str(run_id), None)
    except Exception:
        pass
    return {"ok": True, "updated": True, "arm_id": arm_id, "reward": reward, "n": n1, "mean_reward": m1}


def summarize(state: dict[str, Any], arms: list[RLArm]) -> dict[str, Any]:
    arms_state = state.get("arms") or {}
    out = []
    for a in arms:
        st = arms_state.get(a.arm_id) or {}
        out.append({
            "arm_id": a.arm_id,
            "name": a.name,
            "n": int(st.get("n", 0)),
            "mean_reward": float(st.get("mean_reward", 0.0)),
            "config": a.config,
        })
    out = sorted(out, key=lambda x: (x["mean_reward"], x["n"]), reverse=True)
    return {
        "ok": True,
        "schema_version": int(state.get("schema_version", 1)),
        "total_updates": int(state.get("total_updates", 0)),
        "arms": out,
        "pending_runs": len((state.get("runs") or {})),
        "updated_at_utc": state.get("updated_at_utc", ""),
    }

