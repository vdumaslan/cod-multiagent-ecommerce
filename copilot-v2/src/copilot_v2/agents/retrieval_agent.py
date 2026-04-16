from __future__ import annotations

import json
from pathlib import Path


def _load_replay_samples() -> list[dict[str, object]]:
    docs_path = Path(__file__).resolve().parents[3] / "docs" / "debate_replay_sample.json"
    if not docs_path.is_file():
        return []
    raw = json.loads(docs_path.read_text(encoding="utf-8"))
    return raw.get("samples", []) if isinstance(raw, dict) else []


def _pick_sample(user_query: str, samples: list[dict[str, object]]) -> dict[str, object] | None:
    query_terms = {token for token in user_query.lower().split() if token}
    best_score = -1
    best_sample: dict[str, object] | None = None
    for sample in samples:
        goal = str(sample.get("goal", "")).lower()
        score = sum(1 for term in query_terms if term in goal)
        if score > best_score:
            best_score = score
            best_sample = sample
    return best_sample or (samples[0] if samples else None)


def run_retrieval(*, user_query: str) -> dict[str, object]:
    query = user_query.strip()
    samples = _load_replay_samples()
    selected = _pick_sample(query, samples)
    if not selected:
        return {
            "summary": f"No replay samples found for '{query}'.",
            "top_findings": [],
            "selected_sample": {},
        }

    candidates = selected.get("candidates", [])
    top_findings: list[str] = []
    for candidate in candidates[:3]:
        if not isinstance(candidate, dict):
            continue
        pid = str(candidate.get("product_id", "unknown"))
        score = candidate.get("evidence", {}).get("retrieval_score", "n/a")
        top_findings.append(f"Product {pid} retrieved with score {score}.")

    return {
        "summary": f"Loaded replay sample goal '{selected.get('goal', '')}' for query '{query}'.",
        "top_findings": top_findings,
        "selected_sample": selected,
    }

