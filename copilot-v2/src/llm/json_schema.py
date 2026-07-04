from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SchemaError(Exception):
    message: str

    def __str__(self) -> str:  # pragma: no cover
        return self.message


def _repair_json_text(s: str) -> str:
    """Light repairs for common LLM JSON mistakes (not a full parser)."""
    t = s.strip()
    t = t.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    # Remove trailing commas before } or ]
    t = re.sub(r",(\s*[}\]])", r"\1", t)
    return t


def _balanced_substring(s: str, open_ch: str, close_ch: str) -> str | None:
    """First top-level balanced span from open_ch to matching close_ch, or None."""
    start = s.find(open_ch)
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _try_parse_json_object(s: str) -> dict[str, Any] | None:
    for raw in (s, _repair_json_text(s)):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
                # Judge sometimes returns a bare array of actions.
                return {"ranked_actions": obj}
        except Exception:
            continue
    return None


def extract_json_object(text: str) -> dict[str, Any]:
    s = (text or "").strip()
    if "```" in s:
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1]
            s_lines = s.splitlines()
            if s_lines and s_lines[0].strip().lower() in {"json", "javascript"}:
                s = "\n".join(s_lines[1:])
            s = s.strip()

    out = _try_parse_json_object(s)
    if out is not None:
        return out

    # Balanced brace object (handles extra text; avoids naive rfind on nested braces).
    cand = _balanced_substring(s, "{", "}")
    if cand:
        out = _try_parse_json_object(cand)
        if out is not None:
            return out

    # Balanced array at top level (often judge output).
    cand_a = _balanced_substring(s, "[", "]")
    if cand_a:
        out = _try_parse_json_object(cand_a)
        if out is not None:
            return out

    # Last resort: first { ... } slice (legacy).
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        cand2 = s[start : end + 1]
        out = _try_parse_json_object(cand2)
        if out is not None:
            return out

    raise SchemaError("Could not parse JSON object from LLM output.")


def _ensure_str(x: Any) -> str:
    return "" if x is None else str(x)


def _ensure_float(x: Any, *, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _normalize_ranked_actions_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept common judge variants: ranked_actions, actions, or a single action object."""
    if "ranked_actions" in payload and isinstance(payload.get("ranked_actions"), list):
        return payload
    if "actions" in payload and isinstance(payload.get("actions"), list):
        return {"ranked_actions": list(payload["actions"])}
    # Single action flattened at root
    if payload.get("product_id") and ("action_type" in payload or "recommended_price_change_pct" in payload):
        return {"ranked_actions": [payload]}
    return payload


def validate_ranked_actions(payload: dict[str, Any], *, allowed_product_ids: set[str], top_k: int) -> list[dict[str, Any]]:
    payload = _normalize_ranked_actions_payload(dict(payload))
    if "ranked_actions" not in payload or not isinstance(payload["ranked_actions"], list):
        raise SchemaError("Missing ranked_actions list.")

    out: list[dict[str, Any]] = []
    seen_pid: set[str] = set()
    for item in payload["ranked_actions"]:
        if not isinstance(item, dict):
            continue
        pid = _ensure_str(item.get("product_id")).strip()
        if not pid or pid not in allowed_product_ids:
            continue
        if pid in seen_pid:
            continue
        seen_pid.add(pid)
        action_type = _ensure_str(item.get("action_type")).strip().lower() or "reprice"
        if action_type not in {"reprice", "investigate", "hold", "promote", "restock"}:
            action_type = "reprice"
        rec = _ensure_float(item.get("recommended_price_change_pct"), default=0.0)

        rationale = item.get("rationale_bullets")
        risk = item.get("risk_bullets")
        if not isinstance(rationale, list):
            rationale = []
        if not isinstance(risk, list):
            risk = []

        out.append(
            {
                "product_id": pid,
                "action_type": action_type,
                "recommended_price_change_pct": float(rec),
                "rationale_bullets": [str(x) for x in rationale][:4],
                "risk_bullets": [str(x) for x in risk][:3],
            }
        )
        if len(out) >= int(top_k):
            break
    if not out:
        raise SchemaError("No valid ranked_actions after validation.")
    return out


def validate_specialist_proposal(payload: dict[str, Any], *, allowed_product_ids: set[str], top_k: int) -> dict[str, Any]:
    if not isinstance(payload.get("proposed_actions"), list):
        raise SchemaError("Missing proposed_actions list.")
    acts: list[dict[str, Any]] = []
    seen_pid: set[str] = set()
    for it in payload["proposed_actions"]:
        if not isinstance(it, dict):
            continue
        pid = _ensure_str(it.get("product_id")).strip()
        if not pid or pid not in allowed_product_ids:
            continue
        if pid in seen_pid:
            continue
        seen_pid.add(pid)
        acts.append(
            {
                "product_id": pid,
                "action_type": _ensure_str(it.get("action_type")).strip().lower() or "reprice",
                "recommended_price_change_pct": _ensure_float(it.get("recommended_price_change_pct"), default=0.0),
            }
        )
        if len(acts) >= int(top_k):
            break
    if not acts:
        raise SchemaError("No valid proposed_actions.")

    def _list_str(x: Any, n: int) -> list[str]:
        if not isinstance(x, list):
            return []
        return [str(z) for z in x][:n]

    return {
        "proposed_actions": acts,
        "key_claims": _list_str(payload.get("key_claims"), 5),
        "concerns": _list_str(payload.get("concerns"), 5),
    }


def validate_peer_review(payload: dict[str, Any]) -> dict[str, Any]:
    def _list_str(x: Any, n: int) -> list[str]:
        if not isinstance(x, list):
            return []
        return [str(z) for z in x][:n]

    return {
        "agreements": _list_str(payload.get("agreements"), 6),
        "disagreements": _list_str(payload.get("disagreements"), 6),
        "suggested_changes": _list_str(payload.get("suggested_changes"), 6),
    }

