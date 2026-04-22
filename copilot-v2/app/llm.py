"""Ollama client and JSON extraction utilities — self-contained, no src/ imports."""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OllamaChatResult:
    content: str
    raw: dict[str, Any]
    elapsed_s: float


class OllamaClient:
    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        timeout_s: float = 120.0,
        retries: int = 1,
        backoff_s: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.backoff_s = float(backoff_s)

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.9,
        num_predict: int = 800,
        seed: int = 42,
    ) -> OllamaChatResult:
        url = f"{self.base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": str(model),
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "num_predict": int(num_predict),
                "seed": int(seed),
            },
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        for attempt in range(self.retries + 1):
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw_txt = resp.read().decode("utf-8", errors="replace")
                j = json.loads(raw_txt)
                content = str((j.get("message") or {}).get("content") or "")
                return OllamaChatResult(content=content, raw=j, elapsed_s=float(time.time() - t0))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                if attempt < self.retries:
                    time.sleep(self.backoff_s * (2**attempt))
                    continue
                raise


# ---------------------------------------------------------------------------
# JSON extraction and schema validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SchemaError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def _repair_json(s: str) -> str:
    t = s.strip()
    t = t.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    t = re.sub(r",(\s*[}\]])", r"\1", t)
    return t


def _balanced(s: str, open_ch: str, close_ch: str) -> str | None:
    start = s.find(open_ch)
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
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
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return s[start: i + 1]
    return None


def _try_parse(s: str) -> dict[str, Any] | None:
    for raw in (s, _repair_json(s)):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
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
            lines = s.splitlines()
            if lines and lines[0].strip().lower() in {"json", "javascript"}:
                s = "\n".join(lines[1:])
            s = s.strip()
    out = _try_parse(s)
    if out is not None:
        return out
    for cand in (_balanced(s, "{", "}"), _balanced(s, "[", "]")):
        if cand:
            out = _try_parse(cand)
            if out is not None:
                return out
    start, end = s.find("{"), s.rfind("}")
    if start >= 0 and end > start:
        out = _try_parse(s[start: end + 1])
        if out is not None:
            return out
    raise SchemaError("Could not parse JSON object from LLM output.")


def _s(x: Any) -> str:
    return "" if x is None else str(x)


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _lst(x: Any, n: int) -> list[str]:
    if not isinstance(x, list):
        return []
    return [str(z) for z in x][:n]


def validate_ranked_actions(payload: dict[str, Any], *, allowed: set[str], top_k: int) -> list[dict[str, Any]]:
    if "ranked_actions" not in payload and "actions" in payload:
        payload = {"ranked_actions": payload["actions"]}
    if not isinstance(payload.get("ranked_actions"), list):
        raise SchemaError("Missing ranked_actions list.")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in payload["ranked_actions"]:
        if not isinstance(item, dict):
            continue
        pid = _s(item.get("product_id")).strip()
        if not pid or pid not in allowed or pid in seen:
            continue
        seen.add(pid)
        action_type = _s(item.get("action_type")).strip().lower() or "reprice"
        if action_type not in {"reprice", "investigate", "hold", "promote", "restock"}:
            action_type = "reprice"
        out.append({
            "product_id": pid,
            "action_type": action_type,
            "recommended_price_change_pct": _f(item.get("recommended_price_change_pct")),
            "rationale_bullets": _lst(item.get("rationale_bullets"), 4),
            "risk_bullets": _lst(item.get("risk_bullets"), 3),
        })
        if len(out) >= top_k:
            break
    if not out:
        raise SchemaError("No valid ranked_actions after validation.")
    return out


def validate_specialist_proposal(payload: dict[str, Any], *, allowed: set[str], top_k: int) -> dict[str, Any]:
    if not isinstance(payload.get("proposed_actions"), list):
        raise SchemaError("Missing proposed_actions list.")
    acts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for it in payload["proposed_actions"]:
        if not isinstance(it, dict):
            continue
        pid = _s(it.get("product_id")).strip()
        if not pid or pid not in allowed or pid in seen:
            continue
        seen.add(pid)
        acts.append({
            "product_id": pid,
            "action_type": _s(it.get("action_type")).strip().lower() or "reprice",
            "recommended_price_change_pct": _f(it.get("recommended_price_change_pct")),
        })
        if len(acts) >= top_k:
            break
    if not acts:
        raise SchemaError("No valid proposed_actions.")
    return {
        "proposed_actions": acts,
        "key_claims": _lst(payload.get("key_claims"), 5),
        "concerns": _lst(payload.get("concerns"), 5),
    }


def validate_peer_review(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "agreements": _lst(payload.get("agreements"), 6),
        "disagreements": _lst(payload.get("disagreements"), 6),
        "suggested_changes": _lst(payload.get("suggested_changes"), 6),
    }
