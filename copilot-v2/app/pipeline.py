"""Main entry point: wire retrieval → specialist enrichment → optional ACJ debate."""
from __future__ import annotations

import difflib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agents.retrieval_agent import RetrievalAgent
from app.agents.pricing_agent import PricingAgent
from app.agents.sentiment_agent import SentimentAgent
from app.agents.inventory_agent import InventoryAgent
from app.agents.orchestrator.orchestrator import ACJConfig, run_acj, continue_acj, run_judge_only
from app.llm import OllamaClient, extract_json_object
from app.rl import default_arms, load_state, save_state, select_arm, attach_run

SNAPSHOT_ID = "38710839ca6e1009"


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "__float__"):
        return float(obj)
    if hasattr(obj, "__int__"):
        return int(obj)
    return str(obj)


class Pipeline:
    def __init__(
        self,
        snapshot_id: str = SNAPSHOT_ID,
        artifacts_root: Path | None = None,
        ollama_base_url: str = "http://localhost:11434",
        data_backend: str | None = None,
    ) -> None:
        self.snapshot_id = snapshot_id
        self.data_backend = (data_backend or os.environ.get("COPILOT_DATA_BACKEND") or "local").strip().lower()
        root = artifacts_root or Path(__file__).resolve().parent.parent / "artifacts"

        self.retrieval = RetrievalAgent(snapshot_id=snapshot_id, artifacts_root=root)
        min_score_env = os.environ.get("COPILOT_RETRIEVAL_MIN_SCORE", "").strip()
        try:
            self.retrieval.config.min_score = float(min_score_env) if min_score_env else 0.80
        except Exception:
            self.retrieval.config.min_score = 0.80
        self.retrieval.load_index()

        self.pricing = PricingAgent(snapshot_id=snapshot_id, artifacts_root=root, data_backend=self.data_backend)
        self.sentiment = SentimentAgent(snapshot_id=snapshot_id, artifacts_root=root, data_backend=self.data_backend)
        self.inventory = InventoryAgent(snapshot_id=snapshot_id, artifacts_root=root, data_backend=self.data_backend)

        self.ollama = OllamaClient(base_url=ollama_base_url)
        self._index_meta = str(
            root / "indexes" / snapshot_id / "dense" / "intfloat_e5-large-v2" / "index_meta.json"
        )
        self._runs_root = root / "runs" / snapshot_id
        self._catalog_summary_cache: dict[str, Any] | None = None
        self._catalog_facets_cache: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def catalog_summary(self) -> dict[str, Any]:
        """Return a lightweight snapshot of what's in the catalog for UI guidance."""
        if self._catalog_summary_cache is not None:
            return self._catalog_summary_cache

        # Corpus is loaded by RetrievalAgent.load_index() at init.
        docs = self.retrieval._corpus or []
        cat_counts: dict[str, int] = {}
        sub_counts: dict[str, int] = {}

        def _inc(m: dict[str, int], k: str) -> None:
            k = str(k or "").strip()
            if not k:
                return
            m[k] = int(m.get(k, 0)) + 1

        for d in docs:
            txt = str((d or {}).get("product_document") or "")
            m = re.search(r"^category:\s*(.*)$", txt, flags=re.M)
            if m:
                _inc(cat_counts, m.group(1))
            m2 = re.search(r"^subcategory:\s*(.*)$", txt, flags=re.M)
            if m2:
                _inc(sub_counts, m2.group(1))

        top_categories = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:12]
        top_subcategories = sorted(sub_counts.items(), key=lambda x: x[1], reverse=True)[:12]
        self._catalog_summary_cache = {
            "snapshot_id": self.snapshot_id,
            "n_products": len(docs),
            "top_categories": [{"name": k, "count": v} for k, v in top_categories],
            "top_subcategories": [{"name": k, "count": v} for k, v in top_subcategories],
        }
        return self._catalog_summary_cache

    def catalog_facets(self) -> dict[str, Any]:
        """Return category → subcategory facets (cached) for guided UI filtering."""
        if self._catalog_facets_cache is not None:
            return self._catalog_facets_cache

        docs = self.retrieval._corpus or []
        cat_counts: dict[str, int] = {}
        sub_by_cat: dict[str, dict[str, int]] = {}

        def _get_field(txt: str, key: str) -> str:
            m = re.search(rf"^{re.escape(key)}:\s*(.*)$", txt, flags=re.M)
            return (m.group(1).strip() if m else "")

        for d in docs:
            txt = str((d or {}).get("product_document") or "")
            cat = _get_field(txt, "category")
            sub = _get_field(txt, "subcategory")
            if cat:
                cat_counts[cat] = int(cat_counts.get(cat, 0)) + 1
            if cat and sub:
                sub_by_cat.setdefault(cat, {})
                sub_by_cat[cat][sub] = int(sub_by_cat[cat].get(sub, 0)) + 1

        categories = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)
        # Keep payload reasonably small; UI can still search within these.
        categories = categories[:50]
        subcats_by_category: dict[str, list[dict[str, Any]]] = {}
        for cat, _ in categories:
            subcounts = sub_by_cat.get(cat, {})
            top_subs = sorted(subcounts.items(), key=lambda x: x[1], reverse=True)[:50]
            subcats_by_category[cat] = [{"name": k, "count": v} for k, v in top_subs]

        self._catalog_facets_cache = {
            "snapshot_id": self.snapshot_id,
            "n_products": len(docs),
            "categories": [{"name": k, "count": v} for k, v in categories],
            "subcategories_by_category": subcats_by_category,
        }
        return self._catalog_facets_cache

    def retrieval_preview(
        self,
        *,
        goal: str,
        constraints: dict[str, Any] | None = None,
        top_k_preview: int = 5,
        selected_category: str = "",
        selected_subcategory: str = "",
    ) -> dict[str, Any]:
        """Preview retrieval results (fast) so UI can show match count and examples."""
        constraints = constraints or {}
        # Use the same rewrite behavior as pipeline.
        rewrite = self._rewrite_goal_for_retrieval(
            goal,
            selected_category=selected_category,
            selected_subcategory=selected_subcategory,
        )
        retrieval_query = str(rewrite.get("retrieval_query") or "").strip()
        clarifying_question = str(rewrite.get("clarifying_question") or "").strip() or None

        if rewrite.get("used") and clarifying_question and not retrieval_query:
            return {
                "ok": True,
                "goal": goal,
                "retrieval_query": "",
                "clarifying_question": clarifying_question,
                "min_score": float(self.retrieval.config.min_score or 0.0),
                "n_candidates_above_min_score": 0,
                "top_candidates": [],
            }

        q = retrieval_query or goal
        results = self.retrieval.retrieve(q, top_k=max(1, min(int(top_k_preview), 20)))

        def _parse_field(doc: str, key: str) -> str:
            m = re.search(rf"(?:^|\n){re.escape(key)}:\s*(.+?)(?:\n|$)", doc)
            return m.group(1).strip() if m else ""

        cands = []
        for r in results:
            doc = str(r.get("product_document") or "")
            cands.append({
                "product_id": str(r.get("product_id") or ""),
                "score": float(r.get("retrieval_score") or 0.0),
                "title": (str(r.get("title") or "") or _parse_field(doc, "title"))[:140],
                "category": (str(r.get("category") or "") or _parse_field(doc, "category"))[:80],
                "subcategory": (str(r.get("subcategory") or "") or _parse_field(doc, "subcategory"))[:80],
            })

        return {
            "ok": True,
            "goal": goal,
            "retrieval_query": q,
            "clarifying_question": clarifying_question,
            "min_score": float(self.retrieval.config.min_score or 0.0),
            "n_candidates_above_min_score": len(results),
            "top_candidates": cands,
        }

    def _rewrite_goal_for_retrieval(
        self,
        goal: str,
        *,
        selected_subcategory: str = "",
        selected_category: str = "",
    ) -> dict[str, Any]:
        """Rewrite a broad business goal into a product-oriented retrieval query.

        Priority order (deterministic first, LLM last):
          1. selected_subcategory from UI  →  most specific, use directly
          2. selected_category from UI     →  broader but still product-scoped
          3. heuristic / LLM rewrite       →  last resort for free-text goals

        This keeps retrieval stable for demos and avoids FAISS receiving
        business-strategy language like "safe repricing and targeted promotions".
        """
        goal = str(goal or "").strip()
        original_goal = goal

        # ── Priority 1: UI-selected subcategory ─────────────────────────────────
        sub = str(selected_subcategory or "").strip()
        cat = str(selected_category or "").strip()
        if sub:
            return {
                "ok": True,
                "used": True,
                "original_goal": original_goal,
                "retrieval_query": sub,
                "source": "selected_subcategory",
                "notes": f"Retrieval scoped to selected subcategory: '{sub}'",
            }
        # ── Priority 2: UI-selected category ────────────────────────────────────
        if cat:
            return {
                "ok": True,
                "used": True,
                "original_goal": original_goal,
                "retrieval_query": cat,
                "source": "selected_category",
                "notes": f"Retrieval scoped to selected category: '{cat}'",
            }
        # ── Priority 3: Deterministic "for X" extraction ─────────────────────────
        # Done BEFORE the LLM-enabled flag so it always fires regardless of RL arm config.
        # "reduce returns for kitchen storage products" → "kitchen storage products"
        # "increase sales for Bed Frames with safe repricing" → "bed frames"
        if goal:
            _goal_l_pre = goal.lower()
            _broad_markers_pre = [
                "increase revenue", "increase profit", "increase sales", "increase margin", "increase conversion",
                "grow revenue", "grow sales", "grow profit",
                "boost sales", "boost revenue", "boost profit",
                "improve business", "improve performance", "improve conversion",
                "reduce returns", "reduce churn", "reduce complaints",
                "clear inventory", "avoid stockouts", "prevent stockouts",
            ]
            # Strip any "(scope: ...)" hint before analysis
            _goal_stripped = re.sub(r"\s*\(scope:[^)]*\)", "", _goal_l_pre).strip()
            _has_for = bool(re.search(
                r"\bfor\s+[a-z0-9][a-z0-9&'\\-]{1,}(?:\s+[a-z0-9][a-z0-9&'\\-]{1,}){0,4}\b",
                _goal_stripped,
            ))
            _has_broad = any(m in _goal_stripped for m in _broad_markers_pre)
            if _has_for and _has_broad:
                _STOP = r"(?:with|and|or|while|before|after|using|safely|quickly|carefully|targeted|promotions?|repricing|discounts?)"
                _m = re.search(
                    rf"\bfor\s+((?:(?!{_STOP}\b)[a-z0-9][a-z0-9&'\\-]{{1,}})(?:\s+(?:(?!{_STOP}\b)[a-z0-9][a-z0-9&'\\-]{{1,}})){{0,2}})\b",
                    _goal_stripped,
                )
                if _m:
                    _product_term = _m.group(1).strip()
                    return {
                        "ok": True, "used": True,
                        "original_goal": original_goal,
                        "retrieval_query": _product_term,
                        "source": "extracted_for_phrase",
                        "notes": f"Business goal detected; retrieval scoped to product term: '{_product_term}'",
                    }

        enabled = os.environ.get("COPILOT_ENABLE_QUERY_REWRITE", "1").strip().lower() not in {"0", "false", "no"}
        model = os.environ.get("COPILOT_QUERY_REWRITE_MODEL", "qwen2.5:7b-instruct").strip() or "qwen2.5:7b-instruct"
        if not enabled or not goal:
            return {"ok": True, "used": False, "original_goal": goal, "retrieval_query": goal}

        # Lightweight typo correction for common intent verbs (esp. first token), so broad-goal detection
        # and rewrites don't silently fail on obvious misspellings like "Incxrease ...".
        def _fix_common_intent_typos(s: str) -> tuple[str, list[dict[str, str]]]:
            vocab = [
                "increase", "decrease", "reduce", "avoid", "prevent", "clear", "grow", "boost", "improve",
                "revenue", "profit", "sales", "returns", "stockouts", "conversion", "margin",
                "inventory", "performance",
            ]
            words = re.findall(r"[A-Za-z]{4,}", s)
            if not words:
                return s, []
            fixes: list[dict[str, str]] = []
            fixed = s

            def _adjacent_transposition_match(w: str) -> str | None:
                wl = w.lower()
                for v in vocab:
                    if len(v) != len(wl):
                        continue
                    # Detect a single adjacent swap (Damerau-like) e.g. gorw -> grow, grwo -> grow.
                    mism = [i for i, (a, b) in enumerate(zip(wl, v)) if a != b]
                    if len(mism) != 2:
                        continue
                    i, j = mism
                    if j == i + 1 and wl[i] == v[j] and wl[j] == v[i] and wl[:i] == v[:i] and wl[j + 1 :] == v[j + 1 :]:
                        return v
                return None

            # Only auto-correct the first two words (highest leverage, low risk of mangling product names).
            # This catches cases like "Boost saels ..." and "Clear invetory ...".
            for i in range(min(2, len(words))):
                w = words[i]
                cand = difflib.get_close_matches(w.lower(), vocab, n=1, cutoff=0.80)
                if not cand or cand[0] == w.lower():
                    trans = _adjacent_transposition_match(w)
                    if not trans or trans == w.lower():
                        continue
                    corrected = trans
                else:
                    corrected = cand[0]
                if w[:1].isupper():
                    corrected = corrected[:1].upper() + corrected[1:]
                fixed = re.sub(rf"\b{re.escape(w)}\b", corrected, fixed, count=1)
                fixes.append({"from": w, "to": corrected})

            return fixed, fixes

        goal_fixed, typo_fixes = _fix_common_intent_typos(goal)
        # Strip UI-appended scope hint before broad-goal analysis so it doesn't
        # inflate the word count and bypass the query rewrite.
        goal_for_logic = re.sub(r"\s*\(scope:[^)]*\)", "", goal_fixed).strip()

        # Heuristic: only rewrite when the goal looks broad/strategy-level.
        broad_markers = [
            "increase revenue", "increase profit", "increase sales", "increase margin", "increase conversion",
            "grow revenue", "grow sales", "grow profit",
            "boost sales", "boost revenue", "boost profit",
            "improve business", "improve performance", "improve conversion",
            "reduce returns", "reduce churn", "reduce complaints",
            "clear inventory", "avoid stockouts", "prevent stockouts",
        ]
        goal_l = goal_for_logic.lower()
        # Detect goals that already name a product / category (e.g. "grow revenue for Baking Cups").
        # Use {1,} (≥2 chars total) so short product names like "Bed", "Rug", "Mop" are included.
        has_specific_for_phrase = bool(
            re.search(r"\bfor\s+[a-z0-9][a-z0-9&'\\-]{1,}(?:\s+[a-z0-9][a-z0-9&'\\-]{1,}){0,4}\b", goal_l)
        )
        def _passthrough() -> dict[str, object]:
            if typo_fixes and goal_fixed != original_goal:
                note = "; ".join([f"{f['from']} → {f['to']}" for f in typo_fixes])
                return {"ok": True, "used": True, "original_goal": original_goal, "retrieval_query": goal_fixed, "notes": f"Typo corrected: {note}"}
            return {"ok": True, "used": False, "original_goal": original_goal, "retrieval_query": original_goal}

        # "for X" phrase found — but if the goal also contains broad business markers, the
        # product term (after "for") is the right retrieval query, NOT the full sentence.
        # e.g. "reduce returns for kitchen storage products" → retrieval_query="kitchen storage products"
        # e.g. "grow revenue for Baking Cups" (pure product goal, no jargon) → passthrough
        if has_specific_for_phrase:
            goal_has_broad = any(m in goal_l for m in broad_markers)
            if goal_has_broad:
                # Extract up to 3 clean product-word tokens (no spaces in each token).
                # Stop before common modifier words so "coffee makers safely" → "coffee makers".
                _STOP = r"(?:with|and|or|while|before|after|using|safely|quickly|carefully|targeted|promotions?|repricing|discounts?)"
                m_for = re.search(
                    rf"\bfor\s+((?:(?!{_STOP}\b)[a-z0-9][a-z0-9&'\\-]{{1,}})(?:\s+(?:(?!{_STOP}\b)[a-z0-9][a-z0-9&'\\-]{{1,}})){{0,2}})\b",
                    goal_l,
                )
                if m_for:
                    product_term = m_for.group(1).strip()
                    return {
                        "ok": True, "used": True,
                        "original_goal": original_goal,
                        "retrieval_query": product_term,
                        "source": "extracted_for_phrase",
                        "notes": f"Business goal detected; retrieval scoped to product term: '{product_term}'",
                    }
            # Pure product phrase (no broad jargon) and short enough → passthrough
            if len(goal_for_logic.split()) <= 7:
                return _passthrough()

        is_broad = any(m in goal_l for m in broad_markers) and len(goal_for_logic.split()) <= 20
        if not is_broad:
            return _passthrough()

        messages = [
            {
                "role": "system",
                "content": (
                    "You rewrite vague business goals into product-oriented search queries for an e-commerce catalog. "
                    "Output STRICT JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Rewrite the goal into a retrieval query that would match product titles/descriptions.\n"
                    "Return JSON with keys:\n"
                    "- retrieval_query (string)\n"
                    "- clarifying_question (string, optional)\n"
                    "- notes (string, optional)\n"
                    "Rules:\n"
                    "- Keep retrieval_query short (<= 12 words).\n"
                    "- Prefer concrete product/category terms; avoid business jargon.\n"
                    "- If goal is too vague, set retrieval_query to \"\" and provide a clarifying_question.\n"
                    f"GOAL: {goal_fixed}"
                ),
            },
        ]
        raw_text = ""
        try:
            res = self.ollama.chat(model=model, messages=messages, temperature=0.0, num_predict=200, seed=17)
            raw_text = res.content
            obj = extract_json_object(res.content)
            rq = str(obj.get("retrieval_query") or "").strip()
            cq = str(obj.get("clarifying_question") or "").strip()
            notes = str(obj.get("notes") or "").strip()
            # If rewriter returned both, treat as clarification — do not proceed with retrieval.
            if rq and cq:
                rq = ""
            if not rq and not cq:
                cq = "What product category (or example SKU) are you trying to improve revenue for?"
            if typo_fixes and goal_fixed != original_goal:
                note = "; ".join([f"{f['from']} → {f['to']}" for f in typo_fixes])
                prefix = f"Typo corrected: {note}."
                notes = f"{prefix} {notes}".strip()
            return {
                "ok": True,
                "used": True,
                "original_goal": original_goal,
                "retrieval_query": rq,
                "clarifying_question": cq,
                "notes": notes,
                "raw": raw_text[:1500],
                "model": model,
            }
        except Exception as e:
            return {
                "ok": False,
                "used": False,
                "original_goal": original_goal,
                "retrieval_query": original_goal,
                "error": str(e)[:200],
                "raw": raw_text[:500],
                "model": model,
            }

    @staticmethod
    def _compute_return_rate(total_returns: float, total_units_sold: float) -> float | None:
        """Return rate as a fraction (0..1), or None when no sales data is available."""
        if total_units_sold > 0:
            return float(total_returns) / float(total_units_sold)
        return None

    def _suggest_action(
        self,
        *,
        objective: str,
        inventory_status: str,
        risk_flag: bool,
        pricing_source: str,
        predicted_price_change_pct: float,
        n_reviews: float,
        p_neg: float,
        total_returns: float,
        total_units_sold: float = 0.0,
    ) -> str:
        """Lightweight action playbook derived from runtime signals."""
        obj = (objective or "revenue").strip().lower()
        inv = (inventory_status or "unknown").lower()
        src = (pricing_source or "fallback").strip().lower()
        try:
            n_rev = float(n_reviews or 0.0)
        except Exception:
            n_rev = 0.0
        try:
            pchg = float(predicted_price_change_pct or 0.0)
        except Exception:
            pchg = 0.0

        # Inventory hard signals first
        if inv == "stockout_risk":
            return "restock"
        # low_stock → hold; risk_flag → hold UNLESS the product is overstocked.
        # Overstocked products have risk_flag=True in the precompute (surplus risk),
        # but the right response is investigate/promote, not hold.
        if inv == "low_stock" or (risk_flag and inv != "overstocked"):
            return "hold"

        # Missing pricing coverage: don't pretend we can reprice intelligently.
        if src == "fallback":
            return "investigate"

        # Objective-specific nudges (early exits for objectives that override defaults).
        if obj == "avoid_stockouts":
            return "hold"
        if obj == "reduce_returns" and (total_returns >= 2 or (n_rev >= 8 and p_neg >= 0.20)):
            return "investigate"

        # Return-rate and sentiment quality checks run before any positive-action suggestions.
        # This ensures return/sentiment signals block reprice/promote even for grow_revenue.
        return_rate = self._compute_return_rate(total_returns, total_units_sold)
        if return_rate is not None:
            # Hard investigate: ≥ 5 returns AND ≥ 5% rate.
            if total_returns >= 5 and return_rate >= 0.05:
                return "investigate"
            # Soft caution zone: ≥ 3 returns AND ≥ 3% rate.
            # Returns at this level are unexplained-enough to warrant investigation before repricing,
            # regardless of sentiment — ensures the same signal always produces the same starting action.
            # The LLM judge can still override to "reprice" if evidence is strong; the UI will show
            # the override note so the seller understands why the recommendation changed.
            if total_returns >= 3 and return_rate >= 0.03:
                return "investigate"
        else:
            # No volume data — raw count ≥ 5 as a conservative fallback only.
            if total_returns >= 5:
                return "investigate"

        if n_rev >= 10 and p_neg >= 0.30:
            return "investigate"

        # Overstock: prefer promote unless quality signals (checked above) suggest investigating first.
        # grow_revenue / revenue explicitly prefers promote for overstocked products; for healthy stock
        # the fallthrough to reprice already represents the right default.
        if inv == "overstocked":
            return "promote"

        # For clear-inventory objective, promote/reprice down is often safer than raising price.
        if obj == "clear_inventory" and pchg > 0.0:
            return "promote"

        # grow_revenue with healthy stock and clean signals: explicitly prefer reprice.
        # (This is the same as the default fallthrough but makes the goal→action link reportable.)
        if obj in ("revenue", "grow_revenue") and pricing_source == "cache" and not risk_flag:
            return "reprice"

        return "reprice"

    def _enrich(
        self,
        candidates: list[dict[str, Any]],
        *,
        horizon_days: int,
        enable_pricing: bool,
        enable_sentiment: bool,
        constraints: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        constraints = constraints or {}
        try:
            policy_bound = float(constraints.get("max_abs_price_change_pct", 10.0) or 10.0)
        except Exception:
            policy_bound = 10.0
        policy_bound = max(0.01, float(policy_bound))
        exclude_low_stock = bool(constraints.get("exclude_low_stock", False))
        exclude_stockout_risk = bool(constraints.get("exclude_stockout_risk", False))
        do_not_raise_if_p_neg_above = constraints.get("do_not_raise_if_p_neg_above", None)
        try:
            do_not_raise_if_p_neg_above_f = (
                float(do_not_raise_if_p_neg_above) if do_not_raise_if_p_neg_above is not None else None
            )
        except Exception:
            do_not_raise_if_p_neg_above_f = None

        enable_shrink = os.environ.get("COPILOT_ENABLE_PRICING_SHRINKAGE", "0").strip().lower() in {"1", "true", "yes"}

        enriched = []
        for c in candidates:
            pid = str(c.get("product_id") or "")
            if not pid:
                continue

            # Pricing
            pricing_result = self.pricing.lookup(pid) if enable_pricing else {"found": False}
            pricing_source = "cache" if pricing_result.get("found") else "fallback"

            # Raw TabPFN prediction (directional signal — nearly always ≈ ±7.6% due to tanh saturation).
            price_change_raw = (
                float(pricing_result.get("predicted_price_change_pct") or 0.0)
                if pricing_result.get("found")
                else 0.0
            )

            # Gap-proportional magnitude: use price_percentile_in_subcategory from the cache.
            # The TabPFN model reliably predicts DIRECTION (up/down) but not magnitude.
            # Magnitude is computed from the product's position relative to its subcategory median:
            #   - deviation_from_center = |0.5 - percentile|  (0 at median, 0.5 at extreme)
            #   - base_magnitude = deviation * 20   → 0% at median, 10% at p0/p100
            # This gives continuous, proportional recommendations instead of the saturated ±7.6%.
            #
            # Objective-aware direction override (Fix A):
            # The TabPFN model is trained on historical repricing patterns, not on seller objectives.
            # When the objective explicitly requires a directional bias, we override TabPFN's direction:
            #   clear_inventory  + pct ≥ 0.45: force price DOWN (product is at/above median → reduce to
            #                                   stimulate demand; cutting price is the clearest lever).
            #   avoid_stockouts  + pct ≤ 0.55: force price UP   (recover margin while limiting demand).
            _percentile = pricing_result.get("price_percentile_in_subcategory")
            if pricing_source == "cache" and _percentile is not None:
                try:
                    _pct = float(_percentile)
                    _direction = 1.0 if price_change_raw > 0 else (-1.0 if price_change_raw < 0 else 0.0)
                    _obj_dir = str((constraints or {}).get("objective") or "revenue").strip().lower()
                    if _obj_dir == "clear_inventory" and _pct >= 0.45:
                        _direction = -1.0   # must reduce price to clear stock
                    _deviation = abs(0.5 - _pct)          # 0 at median → 0.5 at extremes
                    _base_magnitude = _deviation * 20.0   # 0% at median → 10% at extremes
                    price_change_raw = _direction * _base_magnitude
                except Exception:
                    pass  # fall back to raw TabPFN value if anything fails

            # Shrinkage trigger: based on the computed magnitude (post percentile-adjustment).
            _trigger_shrink = bool(pricing_result.get("found")) and abs(price_change_raw) >= 0.7 * policy_bound
            pricing_info: dict[str, Any] = {
                "source": pricing_source,
                "price_missing": (pricing_source != "cache"),
                "policy_bound": policy_bound,
                "moderate_delta": False,
                "large_delta": False,
                "near_bound": False,
                "predicted_price_change_pct_raw": price_change_raw if pricing_result.get("found") else None,
                "price_percentile_in_subcategory": _percentile,
                "current_price": pricing_result.get("current_price"),
                "shrinkage_enabled": enable_shrink,
                "shrink_factor": 1.0,
                "shrink_applied": False,
            }

            # Sentiment
            sentiment_result = self.sentiment.lookup(pid) if enable_sentiment else {"found": False}
            sentiment_info = (
                {
                    "n_reviews": sentiment_result.get("n_reviews"),
                    "p_pos": sentiment_result.get("p_pos"),
                    "p_neu": sentiment_result.get("p_neu"),
                    "p_neg": sentiment_result.get("p_neg"),
                }
                if sentiment_result.get("found")
                else {}
            )
            p_neg = float(sentiment_info.get("p_neg") or 0.0)

            # Inventory
            inv = self.inventory.lookup(pid)
            on_hand = float(inv.get("on_hand_units") or 0.0)
            safety = float(inv.get("safety_stock_units") or 0.0)
            available = float(inv.get("available_to_sell") or max(on_hand - safety, 0.0))
            mean_rev = float(inv.get("mean_daily_revenue") or 0.0)
            units_sold = float(inv.get("total_units_sold") or 0.0)
            returns = float(inv.get("total_returns") or 0.0)
            return_rate = self._compute_return_rate(returns, units_sold)

            inventory_info = {
                "stock_status": inv.get("stock_status", "unknown"),
                "risk_flag": bool(inv.get("risk_flag", False)),
                "signals": {
                    "on_hand_units": on_hand,
                    "safety_stock_units": safety,
                    "available_to_sell": available,
                    "mean_daily_revenue": mean_rev,
                    "total_units_sold": units_sold,
                    "total_returns": returns,
                    "return_rate": return_rate,
                },
                "rules_fired": inv.get("rules_fired", []) if "rules_fired" in inv else [],
            }

            stock_status = str(inventory_info.get("stock_status") or "unknown").lower()

            # Evidence from retrieval
            retrieval_score = float(c.get("retrieval_score") or 0.0)
            doc_text = str(c.get("product_document") or "")
            evidence = {
                "retrieval_score": retrieval_score,
                "points": [{"type": "retrieval_doc", "text": doc_text[:400], "score": retrieval_score}],
            }

            signals = {
                "margin_pct": 0.0,
                "available_to_sell": available,
                "mean_daily_revenue": mean_rev,
                "total_units_sold": units_sold,
                "total_returns": returns,
                "return_rate": return_rate,
                "on_hand_units": on_hand,
                "safety_stock_units": safety,
            }

            # Quality multiplier: scale recommendation down when signals suggest caution.
            # Applied BEFORE shrinkage so both adjustments compound conservatively.
            #   clean signals (low p_neg, low returns, healthy stock) → 1.0 (full magnitude)
            #   moderate signals (p_neg 10-20%, return_rate 2-4%)     → 0.65
            #   elevated signals (p_neg ≥20%, return_rate ≥4%, risk)  → 0.35
            price_change = float(price_change_raw or 0.0)
            if pricing_source == "cache" and abs(price_change) > 0:
                _ret_rate_val = return_rate if return_rate is not None else 0.0
                _risk_flag_val = bool(inv.get("risk_flag", False))
                if p_neg >= 0.20 or _ret_rate_val >= 0.04 or _risk_flag_val:
                    price_change *= 0.35
                elif p_neg >= 0.10 or _ret_rate_val >= 0.02:
                    price_change *= 0.65
                # else: quality_mult = 1.0, no change

            # Optional shrinkage for large deltas (post soft-cap). Keep behind a flag for A/B.
            if enable_shrink and pricing_source == "cache" and _trigger_shrink:
                # Shrink when evidence is weak.
                # Defaults are intentionally mild; tune via env vars without code changes.
                # retrieval_score is cosine similarity (0..1). sentiment coverage is a proxy for confidence.
                n_reviews = float(sentiment_info.get("n_reviews") or 0.0)
                try:
                    strong = float(os.environ.get("COPILOT_PRICING_SHRINK_FACTOR_STRONG", "0.90").strip() or 0.90)
                except Exception:
                    strong = 0.90
                try:
                    weak = float(os.environ.get("COPILOT_PRICING_SHRINK_FACTOR_WEAK", "0.80").strip() or 0.80)
                except Exception:
                    weak = 0.80
                try:
                    no_reviews_mult = float(os.environ.get("COPILOT_PRICING_SHRINK_NO_REVIEWS_MULT", "0.95").strip() or 0.95)
                except Exception:
                    no_reviews_mult = 0.95

                # Clamp to a safe range to avoid accidental over-shrink.
                strong = max(0.70, min(1.0, strong))
                weak = max(0.60, min(1.0, weak))
                no_reviews_mult = max(0.80, min(1.0, no_reviews_mult))

                factor = strong if retrieval_score >= 0.88 else weak
                if n_reviews <= 0:
                    factor *= no_reviews_mult
                # Reduce-only for clear-inventory objective (avoid shrinking discounts too much).
                obj = str((constraints or {}).get("objective") or "revenue").strip().lower()
                if obj == "clear_inventory" and price_change < 0:
                    factor = min(1.0, factor + 0.10)
                price_change = float(price_change) * float(factor)
                # Clamp to policy bound.
                price_change = max(-policy_bound, min(policy_bound, price_change))
                pricing_info["shrink_factor"] = float(factor)
                pricing_info["shrink_applied"] = True

            # Recompute delta flags from the FINAL (post-shrink) recommended delta.
            # Using the raw value would overstate risk when shrinkage has already reduced it.
            # Thresholds: moderate >7.5%, large >9.5%, near_bound >10% of policy_bound.
            if pricing_source == "cache":
                abs_final = abs(price_change)
                pricing_info["moderate_delta"] = abs_final > 0.75 * policy_bound
                pricing_info["large_delta"]    = abs_final > 0.95 * policy_bound
                pricing_info["near_bound"]     = abs_final > 1.00 * policy_bound

            suggested_action = self._suggest_action(
                objective=str((constraints or {}).get("objective") or "revenue"),
                inventory_status=str(inventory_info.get("stock_status") or "unknown"),
                risk_flag=bool(inventory_info.get("risk_flag", False)),
                pricing_source=str(pricing_info.get("source") or "fallback"),
                predicted_price_change_pct=price_change,
                n_reviews=float(sentiment_info.get("n_reviews") or 0.0),
                p_neg=p_neg,
                total_returns=returns,
                total_units_sold=units_sold,
            )

            # Constraint filters / guardrails (pre-LLM)
            # These keep obviously-invalid repricing recommendations from reaching the debate.
            if exclude_stockout_risk and stock_status == "stockout_risk":
                suggested_action = "restock"
                price_change = 0.0
            if exclude_low_stock and stock_status == "low_stock":
                suggested_action = "hold"
                price_change = 0.0
            if do_not_raise_if_p_neg_above_f is not None and p_neg > do_not_raise_if_p_neg_above_f and price_change > 0.0:
                price_change = 0.0

            enriched.append({
                "product_id": pid,
                "action_type": suggested_action,   # start from playbook; ACJ may override
                "suggested_action": suggested_action,
                "recommended_price_change_pct": price_change,
                "pricing": pricing_info,
                "sentiment": sentiment_info,
                "signals": signals,
                "evidence": evidence,
                "horizon_days": horizon_days,
                "inventory": inventory_info,
            })

        return enriched

    @staticmethod
    @staticmethod
    def _baseline_score(c: dict[str, Any], objective: str = "revenue") -> float:
        """Heuristic score for baseline ranking when the Judge is unavailable.

        Starts from retrieval similarity (0..1) and adjusts for signal quality.
        Scoring is objective-aware:
          - Default (revenue/reprice): reward clean signals, penalise risk.
          - reduce_returns: reward high p_neg and return rate — these products most
            need investigation and should rank highest for a complaints-reduction goal.
        """
        score = float((c.get("evidence") or {}).get("retrieval_score") or 0.0)
        pr = c.get("pricing") or {}
        sent = c.get("sentiment") or {}
        inv = c.get("inventory") or {}
        sig = c.get("signals") or {}

        obj = (objective or "revenue").strip().lower()
        p_neg = float(sent.get("p_neg") or 0.0)
        ret_rate_raw = sig.get("return_rate")
        ret_rate = float(ret_rate_raw) if ret_rate_raw is not None else None
        total_returns = float(sig.get("total_returns") or 0.0)
        stock_st = str(inv.get("stock_status") or "unknown").lower()

        if obj == "reduce_returns":
            # For reduce_returns the goal is to surface the most problematic SKUs first.
            # High negative sentiment and high return rate are POSITIVE ranking signals.
            if p_neg >= 0.30:
                score += 0.30
            elif p_neg >= 0.20:
                score += 0.15
            if ret_rate is not None and ret_rate >= 0.05:
                score += 0.25
            elif ret_rate is not None and ret_rate >= 0.03:
                score += 0.10
            elif total_returns >= 5:
                score += 0.10
            # Pricing availability is neutral for reduce_returns (investigation doesn't require it).
        else:
            # Default: reward clean signals, penalise risk.
            if str(pr.get("source") or "") == "cache":
                score += 0.10               # pricing signal available → small boost
            if pr.get("price_missing"):
                score -= 0.20               # no pricing data → penalise
            if p_neg >= 0.30:
                score -= 0.20               # high negative sentiment → penalise
            if ret_rate is not None and ret_rate >= 0.05:
                score -= 0.15               # elevated return rate → penalise
            elif total_returns >= 5:
                score -= 0.10               # raw count fallback when no rate available

        # Inventory risk adjustments apply regardless of objective.
        if inv.get("risk_flag") and stock_st != "overstocked":
            score -= 0.20               # low_stock / stockout risk_flag → penalise strongly
        elif stock_st == "overstocked":
            score -= 0.05               # overstocked → slight penalty (still actionable via promote)

        return score

    def _make_baseline(
        self, enriched: list[dict[str, Any]], top_n: int, objective: str = "revenue"
    ) -> list[dict[str, Any]]:
        sorted_cands = sorted(
            enriched,
            key=lambda c: self._baseline_score(c, objective=objective),
            reverse=True,
        )
        baseline = []
        for i, c in enumerate(sorted_cands[:top_n]):
            baseline.append({**c, "rank": i + 1, "llm_rationale_bullets": [], "llm_risk_bullets": []})
        return baseline

    def _merge_judge_output(
        self,
        judge_actions: list[dict[str, Any]],
        enriched_by_pid: dict[str, dict[str, Any]],
        horizon_days: int,
        constraints: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        def _deterministic_rationale(base: dict[str, Any]) -> list[str]:
            pr = base.get("pricing") or {}
            sent = base.get("sentiment") or {}
            inv = base.get("inventory") or {}
            sig = base.get("signals") or {}
            ev = base.get("evidence") or {}

            out: list[str] = []
            src = str(pr.get("source") or "unknown")
            if pr.get("price_missing") or src == "fallback":
                out.append("Pricing unavailable for this SKU (price missing/unknown in cache).")
            else:
                pct = float(base.get("recommended_price_change_pct") or 0.0)
                if pr.get("near_bound"):
                    out.append(f"Pricing delta is near policy bound (delta={pct:.2f}%). Treat as lower-trust.")
                elif pr.get("large_delta"):
                    out.append(f"Pricing delta is large vs policy bound (delta={pct:.2f}%). Treat as lower-trust.")
                elif abs(pct) >= 1e-6:
                    out.append(f"Pricing signal suggests delta={pct:.2f}%.")

            stock = str(inv.get("stock_status") or "unknown")
            risk = bool(inv.get("risk_flag", False))
            if stock not in {"unknown", "healthy"}:
                out.append(f"Inventory status={stock}{' (risk_flag=true)' if risk else ''}.")
            elif risk:
                out.append("Inventory risk_flag=true.")

            n_reviews = float(sent.get("n_reviews") or 0.0)
            p_neg = float(sent.get("p_neg") or 0.0)
            if n_reviews > 0:
                out.append(f"Sentiment p_neg={p_neg:.2f} (n_reviews={int(n_reviews)}).")
            returns = float(sig.get("total_returns") or 0.0)
            ret_sold = float(sig.get("total_units_sold") or 0.0)
            if returns > 0:
                if ret_sold > 0:
                    out.append(f"Returns: {int(returns)} of {int(ret_sold)} sold ({returns/ret_sold*100:.1f}% return rate).")
                else:
                    out.append(f"Returns: {int(returns)} units (sales volume unknown).")

            r = float(ev.get("retrieval_score") or 0.0)
            out.append(f"Retrieval similarity={round(r*100):d}%.")

            # De-dup and cap.
            dedup = []
            seen = set()
            for b in out:
                if b not in seen:
                    seen.add(b)
                    dedup.append(b)
            return dedup[:4]

        def _deterministic_risk(base: dict[str, Any]) -> list[str]:
            """Generate risk bullets grounded in actual enrichment signals, never from LLM."""
            pr = base.get("pricing") or {}
            sent = base.get("sentiment") or {}
            inv = base.get("inventory") or {}
            sig = base.get("signals") or {}

            risks: list[str] = []
            stock = str(inv.get("stock_status") or "unknown")
            risk_flag = bool(inv.get("risk_flag", False))
            if stock == "stockout_risk":
                risks.append("Inventory stockout risk: demand-increasing actions may worsen supply gap.")
            elif stock == "low_stock" or risk_flag:
                risks.append(f"Low stock / inventory risk: avoid actions that increase demand{' (risk_flag=true)' if risk_flag else ''}.")
            elif stock == "overstocked":
                risks.append("Overstocked: margin may compress if price is increased without clearing surplus.")

            p_neg = float(sent.get("p_neg") or 0.0)
            n_reviews = float(sent.get("n_reviews") or 0.0)
            if n_reviews == 0:
                risks.append("No review coverage: sentiment confidence is low; treat recommendations with caution.")
            elif p_neg >= 0.30:
                risks.append(f"High negative sentiment ({p_neg:.0%} of {int(n_reviews)} reviews): price increases may accelerate churn.")
            elif p_neg >= 0.20:
                risks.append(f"Moderate negative feedback ({p_neg:.0%}): monitor reviews before aggressive repricing.")

            returns = float(sig.get("total_returns") or 0.0)
            ret_units = float(sig.get("total_units_sold") or 0.0)
            ret_rate = self._compute_return_rate(returns, ret_units)
            if ret_rate is not None:
                rate_pct = ret_rate * 100
                if returns >= 5 and ret_rate >= 0.05:
                    risks.append(f"Elevated return rate ({rate_pct:.1f}% of {int(ret_units)} sold): investigate root cause before changing price.")
                elif ret_rate >= 0.03:
                    risks.append(f"Moderate return rate ({rate_pct:.1f}% of {int(ret_units)} sold): apply carefully and monitor after any change.")
            else:
                if returns >= 5:
                    risks.append(f"Elevated returns ({int(returns)} units, volume unknown): investigate root cause before changing price.")
                elif returns >= 3:
                    risks.append(f"Notable returns ({int(returns)} units): quality risk possible; verify before repricing.")

            if pr.get("price_missing") or str(pr.get("source") or "") == "fallback":
                risks.append("Pricing signal unavailable: recommended price change is low-confidence.")
            elif pr.get("near_bound"):
                risks.append("Suggested delta is near the policy bound — consider a staged rollout.")
            elif pr.get("large_delta"):
                risks.append("Large pricing delta suggested — human review recommended before full rollout.")
            elif pr.get("moderate_delta"):
                risks.append("Moderate pricing delta suggested — apply carefully and monitor response.")

            # De-dup and cap.
            dedup: list[str] = []
            seen: set[str] = set()
            for b in risks:
                if b not in seen:
                    seen.add(b)
                    dedup.append(b)
            return dedup[:3]

        def _looks_generic(bullets: list[str]) -> bool:
            if not bullets:
                return True
            joined = " ".join([str(b or "") for b in bullets]).lower()
            generic_markers = [
                "competitive pricing", "steady demand", "market conditions", "room for minor",
                "features are well-received", "boost demand", "price reduction might",
            ]
            has_marker = any(m in joined for m in generic_markers)
            has_numbers = any(ch.isdigit() for ch in joined)
            # If it's full of generic markers and lacks any numeric grounding, treat as generic.
            return bool(has_marker and not has_numbers)

        out = []
        for i, ja in enumerate(judge_actions):
            pid = str(ja.get("product_id") or "")
            base = enriched_by_pid.get(pid, {})
            llm_rationale = list(ja.get("rationale_bullets") or [])
            llm_risk = list(ja.get("risk_bullets") or [])
            det = _deterministic_rationale(base)
            det_risk = _deterministic_risk(base)
            # Always use deterministic bullets to guarantee factual accuracy.
            # The LLM's action_type and reasoning are preserved; only the cited field values
            # are grounded from actual input data (same principle as RAG — ground citations,
            # not reasoning). LLM outputs are kept as trace-only labels.
            llm_rationale = det
            llm_risk = det_risk

            action_type = str(ja.get("action_type") or "reprice")
            price_change = float(ja.get("recommended_price_change_pct") or 0.0)

            # Hard guardrail 1: if pricing data is unavailable, the judge cannot
            # legitimately recommend a reprice — force investigate + zero delta.
            pricing_info = base.get("pricing") or {}
            if pricing_info.get("price_missing") or str(pricing_info.get("source") or "") == "fallback":
                if action_type == "reprice":
                    action_type = "investigate"
                    price_change = 0.0

            # Hard guardrail 2: mirror the playbook's sentiment/returns threshold.
            # p_neg >= 0.30 with >= 10 reviews → investigate (strong feedback signal).
            # Returns: use dual condition (count + rate) to avoid penalising high-volume products.
            #   total_returns >= 5 AND return_rate >= 5% → investigate (true quality issue).
            #   return_rate >= 3% alone → not a hard block; handled as caution by delta flags / UI.
            sentiment_info = base.get("sentiment") or {}
            sig_info = base.get("signals") or {}
            inv_info = base.get("inventory") or {}
            _p_neg = float(sentiment_info.get("p_neg") or 0.0)
            _n_rev = float(sentiment_info.get("n_reviews") or 0.0)
            _returns = float(sig_info.get("total_returns") or 0.0)
            _units_sold = float(sig_info.get("total_units_sold") or 0.0)
            _return_rate = self._compute_return_rate(_returns, _units_sold)
            if action_type == "reprice" and price_change > 0.0:
                if _n_rev >= 10 and _p_neg >= 0.30:
                    action_type = "investigate"
                    price_change = 0.0
                elif _return_rate is not None and _returns >= 5 and _return_rate >= 0.05:
                    action_type = "investigate"
                    price_change = 0.0
                elif _return_rate is None and _returns >= 5:
                    # No volume denominator; fall back to raw count as weak blocker only when
                    # high enough to be unambiguously risky even without volume context.
                    action_type = "investigate"
                    price_change = 0.0

            # Hard guardrail 3 (reverse): revert a judge-imposed "investigate" back to
            # the playbook's "reprice" when ALL actual enriched signals are clean.
            # This catches LLM hallucination where the judge fabricates high p_neg / returns
            # that don't exist in the real data.  Conditions for revert:
            #   - Judge output is "investigate" but the deterministic playbook said "reprice"
            #   - Pricing is from cache (not fallback)
            #   - return_rate < 3% AND raw returns < 3  (below Option A soft caution zone)
            #   - p_neg < 0.20  (no meaningful negative-sentiment signal)
            #   - inventory is healthy and risk_flag is false
            _playbook_action = str(base.get("suggested_action") or "")
            _stock = str(inv_info.get("stock_status") or "unknown").lower()
            _risk_flag = bool(inv_info.get("risk_flag", False))
            _pricing_src = str(pricing_info.get("source") or "fallback")
            if (
                action_type == "investigate"
                and _playbook_action == "reprice"
                and _pricing_src == "cache"
                and (_return_rate is None or _return_rate < 0.03)
                and _returns < 3
                and _p_neg < 0.20
                and _stock == "healthy"
                and not _risk_flag
            ):
                action_type = "reprice"
                price_change = float(base.get("recommended_price_change_pct") or 0.0)

            # Hard guardrail 4: block anti-objective price increases under clear_inventory.
            # If the merged recommendation is "reprice" with a POSITIVE price change but the
            # objective is to clear inventory, the judge has hallucinated or contradicted the
            # business goal.  Revert to the playbook's (now direction-corrected) price change.
            _obj_g4 = str((constraints or {}).get("objective") or "").strip().lower()
            if action_type == "reprice" and price_change > 0.0 and _obj_g4 == "clear_inventory":
                price_change = float(base.get("recommended_price_change_pct") or 0.0)
                # If playbook recommended a promotion (pre-Fix-A behaviour), honour that.
                if _playbook_action == "promote":
                    action_type = "promote"
                    price_change = 0.0

            # Hard guardrail 5: for reduce_returns, block downgrading investigate → hold.
            # When the objective is to reduce complaints, "hold" on a high-negativity or
            # high-return product means ignoring the signal entirely — wrong for the seller.
            # If the playbook said "investigate" and signals support it, keep "investigate".
            _obj_g5 = str((constraints or {}).get("objective") or "").strip().lower()
            if (
                _obj_g5 == "reduce_returns"
                and action_type == "hold"
                and _playbook_action == "investigate"
                and (_p_neg >= 0.20 or (_return_rate is not None and _return_rate >= 0.03))
            ):
                action_type = "investigate"
                price_change = 0.0

            out.append({
                **base,
                "product_id": pid,
                "action_type": action_type,
                "recommended_price_change_pct": price_change,
                "horizon_days": horizon_days,
                "rank": i + 1,
                "llm_rationale_bullets": llm_rationale,
                "llm_risk_bullets": llm_risk,
            })
        return out

    def _write_stage(self, run_dir: Path, filename: str, data: Any) -> None:
        (run_dir / filename).write_text(
            json.dumps(data, indent=2, default=_json_default), encoding="utf-8"
        )

    def _apply_rl_policy_for_run(
        self,
        *,
        run_id: str,
        owner_id: str,
        constraints: dict[str, Any],
        goal: str,
    ) -> tuple[dict[str, Any], dict[str, str | float | None]]:
        """Select an RL arm and apply its knob overrides for this run.

        Returns (trace_rl_info, restore_snapshot).
        """
        arms = default_arms()
        st = load_state(arms)
        arm = select_arm(arms=arms, state=st)
        attach_run(st, run_id=run_id, arm_id=arm.arm_id)
        save_state(st)

        # Snapshot current globals we mutate.
        restore = {
            "prev_enable_query_rewrite": os.environ.get("COPILOT_ENABLE_QUERY_REWRITE", None),
            "prev_enable_pricing_shrinkage": os.environ.get("COPILOT_ENABLE_PRICING_SHRINKAGE", None),
            "prev_retrieval_min_score": float(self.retrieval.config.min_score or 0.0),
        }

        cfg = arm.config or {}
        # Apply: retrieval min score
        if cfg.get("retrieval_min_score") is not None:
            try:
                self.retrieval.config.min_score = float(cfg["retrieval_min_score"])
            except Exception:
                pass
        # Apply: query rewrite enable
        if cfg.get("enable_query_rewrite") is not None:
            os.environ["COPILOT_ENABLE_QUERY_REWRITE"] = "1" if bool(cfg["enable_query_rewrite"]) else "0"
        # Apply: pricing shrinkage enable
        if cfg.get("enable_pricing_shrinkage") is not None:
            os.environ["COPILOT_ENABLE_PRICING_SHRINKAGE"] = "1" if bool(cfg["enable_pricing_shrinkage"]) else "0"

        return (
            {
                "enabled": True,
                "arm_id": arm.arm_id,
                "arm_name": arm.name,
                "config": cfg,
                "applied": {
                    "retrieval_min_score": float(self.retrieval.config.min_score or 0.0),
                    "enable_query_rewrite": os.environ.get("COPILOT_ENABLE_QUERY_REWRITE", "1"),
                    "enable_pricing_shrinkage": os.environ.get("COPILOT_ENABLE_PRICING_SHRINKAGE", "0"),
                },
                "run_id": run_id,
                "owner_id": owner_id,
                "goal": goal[:200],
                "objective": (constraints or {}).get("objective", "revenue"),
            },
            restore,
        )

    def _restore_rl_mutations(self, restore: dict[str, str | float | None]) -> None:
        try:
            self.retrieval.config.min_score = float(restore.get("prev_retrieval_min_score") or 0.80)
        except Exception:
            pass

        prev_qr = restore.get("prev_enable_query_rewrite")
        if prev_qr is None:
            os.environ.pop("COPILOT_ENABLE_QUERY_REWRITE", None)
        else:
            os.environ["COPILOT_ENABLE_QUERY_REWRITE"] = str(prev_qr)

        prev_sh = restore.get("prev_enable_pricing_shrinkage")
        if prev_sh is None:
            os.environ.pop("COPILOT_ENABLE_PRICING_SHRINKAGE", None)
        else:
            os.environ["COPILOT_ENABLE_PRICING_SHRINKAGE"] = str(prev_sh)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        *,
        goal: str,
        owner_id: str,
        horizon_days: int = 7,
        top_n_actions: int = 3,
        constraints: dict[str, Any] | None = None,
        enable_pricing: bool = True,
        enable_sentiment: bool = True,
        selected_category: str = "",
        selected_subcategory: str = "",
    ) -> dict[str, Any]:
        """Fast path: retrieval + enrichment + baseline only. No LLM calls."""
        constraints = constraints or {"max_abs_price_change_pct": 10.0}
        ts = datetime.now(timezone.utc)
        owner_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", owner_id)[:16]
        run_id = f"{ts.strftime('%Y%m%d_%H%M%S')}_{owner_slug}"
        run_dir = self._runs_root / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        rl_restore: dict[str, str | float | None] | None = None
        rl_trace: dict[str, Any] = {"enabled": False}
        try:
            # Apply RL policy for Version B runs (pipeline is the Version B entrypoint).
            rl_trace, rl_restore = self._apply_rl_policy_for_run(
                run_id=run_id, owner_id=owner_id, constraints=constraints, goal=goal
            )
        except Exception:
            rl_restore = None
            rl_trace = {"enabled": False, "error": "rl_select_failed"}

        try:
            rewrite = self._rewrite_goal_for_retrieval(
                goal,
                selected_category=selected_category,
                selected_subcategory=selected_subcategory,
            )
            self._write_stage(run_dir, "0_query_rewrite.json", rewrite)
            retrieval_query = str(rewrite.get("retrieval_query") or "").strip() or goal
            candidates_raw = self.retrieval.retrieve(retrieval_query)
            self._write_stage(run_dir, "1_retrieval.json", candidates_raw)

            enriched = self._enrich(
                candidates_raw,
                horizon_days=horizon_days,
                enable_pricing=enable_pricing,
                enable_sentiment=enable_sentiment,
                constraints=constraints,
            )
            self._write_stage(run_dir, "2_enriched.json", enriched)

            baseline_for_judge = [{**c, "recommended_price_change_pct": 0.0} for c in enriched]
            _obj_bl = str((constraints or {}).get("objective") or "revenue").strip().lower()
            baseline_ranked = self._make_baseline(baseline_for_judge, top_n_actions, objective=_obj_bl)
            self._write_stage(run_dir, "3_baseline.json", baseline_ranked)

            trace = {
                "snapshot_id": self.snapshot_id,
                "owner_id": owner_id,
                "retrieval_index_meta": self._index_meta,
                "query_rewrite": {k: rewrite.get(k) for k in ("used", "retrieval_query", "clarifying_question", "notes")},
                "pricing": {"wired": enable_pricing, "source": self.pricing.cache_source},
                "sentiment": {"wired": enable_sentiment, "source": self.sentiment.cache_source},
                "inventory": {"wired": True, "mode": "rules_v1", "source": self.inventory.cache_source},
                "demo_allowlist_size": 0,
                "rl": rl_trace,
            }

            return {
                "ok": True,
                "run_id": run_id,
                "goal": goal,
                "enriched_candidates": enriched,
                "baseline_ranked_actions": baseline_ranked,
                "trace": trace,
            }
        finally:
            if rl_restore is not None:
                self._restore_rl_mutations(rl_restore)

    def start_debate(
        self,
        *,
        goal: str,
        owner_id: str,
        constraints: dict[str, Any] | None = None,
        enriched_candidates: list[dict[str, Any]],
        baseline_actions: list[dict[str, Any]],
        top_n_actions: int = 3,
        advocate_model: str = "llama3.1:8b",
        critic_model: str = "qwen2.5:7b-instruct",
        judge_model: str = "qwen2.5:7b-instruct",
        prompt_style: str = "few_shot_json",
        prompt_version: str = "v1",
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Advocate + Critic round 1 on pre-enriched candidates. No retrieval."""
        constraints = constraints or {"max_abs_price_change_pct": 10.0}
        if run_id:
            run_dir = self._runs_root / f"run_{run_id}"
        else:
            ts = datetime.now(timezone.utc)
            owner_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", owner_id)[:16]
            run_id = f"{ts.strftime('%Y%m%d_%H%M%S')}_{owner_slug}_debate"
            run_dir = self._runs_root / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg = ACJConfig(
            advocate_model=advocate_model,
            critic_model=critic_model,
            judge_model=judge_model,
            prompt_style=prompt_style,
            prompt_version=prompt_version,
            top_k=top_n_actions,
        )

        adv_result: dict[str, Any] = {}
        crit_result: dict[str, Any] = {}
        debate_trace = None
        try:
            adv_result, crit_result, debate_trace = run_acj(
                self.ollama,
                cfg=cfg,
                goal=goal,
                constraints=constraints,
                candidates=enriched_candidates,
                baseline_actions=baseline_actions,
                run_dir=run_dir,
            )
        except Exception:
            pass

        return {
            "ok": True,
            "advocate": adv_result,
            "critic": crit_result,
            "debate_trace": debate_trace,
        }

    def run(
        self,
        *,
        goal: str,
        owner_id: str,
        horizon_days: int = 7,
        top_n_actions: int = 3,
        constraints: dict[str, Any] | None = None,
        use_llm_policy: bool = True,
        enable_pricing: bool = True,
        enable_sentiment: bool = True,
        advocate_model: str = "llama3.1:8b",
        critic_model: str = "qwen2.5:7b-instruct",
        judge_model: str = "qwen2.5:7b-instruct",
        prompt_style: str = "few_shot_json",
        prompt_version: str = "v1",
        human_review_mode: str = "skip",
        human_feedback: str | None = None,
        selected_category: str = "",
        selected_subcategory: str = "",
    ) -> dict[str, Any]:
        constraints = constraints or {"max_abs_price_change_pct": 10.0}
        ts = datetime.now(timezone.utc)
        owner_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", owner_id)[:16]
        run_id = f"{ts.strftime('%Y%m%d_%H%M%S')}_{owner_slug}"

        run_dir = self._runs_root / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "run_id": run_id,
            "timestamp_utc": ts.isoformat(),
            "owner_id": owner_id,
            "goal": goal,
            "horizon_days": horizon_days,
            "top_n_actions": top_n_actions,
            "constraints": constraints,
            "use_llm_policy": use_llm_policy,
            "enable_pricing": enable_pricing,
            "enable_sentiment": enable_sentiment,
            "advocate_model": advocate_model,
            "critic_model": critic_model,
            "judge_model": judge_model,
            "prompt_style": prompt_style,
            "prompt_version": prompt_version,
            "human_review_mode": human_review_mode,
            "human_feedback": human_feedback,
            "ab_mode": "B",
        }
        self._write_stage(run_dir, "0_config.json", config)

        rl_restore: dict[str, str | float | None] | None = None
        rl_trace: dict[str, Any] = {"enabled": False}
        try:
            rl_trace, rl_restore = self._apply_rl_policy_for_run(
                run_id=run_id, owner_id=owner_id, constraints=constraints, goal=goal
            )
        except Exception:
            rl_restore = None
            rl_trace = {"enabled": False, "error": "rl_select_failed"}

        # 1. Retrieve candidates
        rewrite = self._rewrite_goal_for_retrieval(
            goal,
            selected_category=selected_category,
            selected_subcategory=selected_subcategory,
        )
        self._write_stage(run_dir, "0_query_rewrite.json", rewrite)
        retrieval_query = str(rewrite.get("retrieval_query") or "").strip() or goal
        candidates_raw = self.retrieval.retrieve(retrieval_query)
        self._write_stage(run_dir, "1_retrieval.json", candidates_raw)

        # 2. Enrich with specialist signals
        enriched = self._enrich(
            candidates_raw,
            horizon_days=horizon_days,
            enable_pricing=enable_pricing,
            enable_sentiment=enable_sentiment,
            constraints=constraints,
        )
        enriched_by_pid = {c["product_id"]: c for c in enriched}
        self._write_stage(run_dir, "2_enriched.json", enriched)

        # 3. Baseline ranked actions (retrieval-score order, price change = 0)
        baseline_for_judge = [
            {**c, "recommended_price_change_pct": 0.0} for c in enriched
        ]
        _obj_bl = str((constraints or {}).get("objective") or "revenue").strip().lower()
        baseline_ranked = self._make_baseline(baseline_for_judge, top_n_actions, objective=_obj_bl)
        self._write_stage(run_dir, "3_baseline.json", baseline_ranked)

        # 4. Round 1: Advocate + Critic (judge runs later via /debate/judge)
        debate_trace: dict[str, Any] | None = None
        if use_llm_policy:
            cfg = ACJConfig(
                advocate_model=advocate_model,
                critic_model=critic_model,
                judge_model=judge_model,
                prompt_style=prompt_style,
                prompt_version=prompt_version,
                top_k=top_n_actions,
            )
            try:
                adv_result, crit_result, debate_trace = run_acj(
                    self.ollama,
                    cfg=cfg,
                    goal=goal,
                    constraints=constraints,
                    candidates=enriched,
                    baseline_actions=baseline_ranked,
                    run_dir=run_dir,
                )
            except Exception:
                pass

        # 5. Build trace
        trace = {
            "snapshot_id": self.snapshot_id,
            "owner_id": owner_id,
            "retrieval_index_meta": self._index_meta,
            "query_rewrite": {k: rewrite.get(k) for k in ("used", "retrieval_query", "clarifying_question", "notes")},
            "pricing": {"wired": enable_pricing, "source": self.pricing.cache_source},
            "sentiment": {"wired": enable_sentiment, "source": self.sentiment.cache_source},
            "inventory": {"wired": True, "mode": "rules_v1", "source": self.inventory.cache_source},
            "demo_allowlist_size": 0,
            "rl": rl_trace,
        }

        # ranked_actions are baseline until the judge runs via /debate/judge
        response = {
            "ok": True,
            "goal": goal,
            "served_from_warm_process": True,
            "input": {
                "goal": goal,
                "horizon_days": horizon_days,
                "constraints": constraints,
            },
            "ranked_actions": baseline_ranked,
            "baseline_ranked_actions": baseline_ranked,
            "enriched_candidates": enriched,
            "trace": trace,
            "debate_trace": debate_trace,
        }

        try:
            return response
        finally:
            if rl_restore is not None:
                self._restore_rl_mutations(rl_restore)

    def continue_debate(
        self,
        *,
        goal: str,
        owner_id: str,
        constraints: dict[str, Any] | None = None,
        enriched_candidates: list[dict[str, Any]],
        baseline_actions: list[dict[str, Any]],
        prev_advocate: dict[str, Any],
        prev_critic: dict[str, Any],
        top_n_actions: int = 3,
        advocate_model: str = "llama3.1:8b",
        critic_model: str = "qwen2.5:7b-instruct",
        judge_model: str = "qwen2.5:7b-instruct",
        prompt_style: str = "few_shot_json",
        prompt_version: str = "v1",
        human_feedback: str | None = None,
    ) -> dict[str, Any]:
        """Run one more Advocate + Critic round. Returns advocate + critic only — no judge."""
        constraints = constraints or {"max_abs_price_change_pct": 10.0}

        ts = datetime.now(timezone.utc)
        owner_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", owner_id)[:16]
        run_id = f"{ts.strftime('%Y%m%d_%H%M%S')}_{owner_slug}_cont"
        run_dir = self._runs_root / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg = ACJConfig(
            advocate_model=advocate_model,
            critic_model=critic_model,
            judge_model=judge_model,
            prompt_style=prompt_style,
            prompt_version=prompt_version,
            top_k=top_n_actions,
        )

        adv_result, crit_result, raw = continue_acj(
            self.ollama,
            cfg=cfg,
            goal=goal,
            constraints=constraints,
            candidates=enriched_candidates,
            baseline_actions=baseline_actions,
            prev_advocate=prev_advocate,
            prev_critic=prev_critic,
            human_feedback=human_feedback,
            run_dir=run_dir,
        )

        return {
            "ok": True,
            "advocate": adv_result,
            "critic": crit_result,
            "raw": raw,
        }

    def run_judge(
        self,
        *,
        goal: str,
        owner_id: str,
        constraints: dict[str, Any] | None = None,
        enriched_candidates: list[dict[str, Any]],
        baseline_actions: list[dict[str, Any]],
        latest_advocate: dict[str, Any],
        latest_critic: dict[str, Any],
        top_n_actions: int = 3,
        judge_model: str = "qwen2.5:7b-instruct",
        prompt_style: str = "few_shot_json",
        prompt_version: str = "v1",
        human_feedback: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Run the judge exactly once on the final advocate + critic outputs."""
        constraints = constraints or {"max_abs_price_change_pct": 10.0}
        enriched_by_pid = {c["product_id"]: c for c in enriched_candidates}
        horizon_days = int(enriched_candidates[0].get("horizon_days", 7)) if enriched_candidates else 7

        if run_id:
            run_dir = self._runs_root / f"run_{run_id}"
        else:
            ts = datetime.now(timezone.utc)
            owner_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", owner_id)[:16]
            run_id = f"{ts.strftime('%Y%m%d_%H%M%S')}_{owner_slug}_judge"
            run_dir = self._runs_root / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg = ACJConfig(
            judge_model=judge_model,
            prompt_style=prompt_style,
            prompt_version=prompt_version,
            top_k=top_n_actions,
        )

        final_actions, judge_raw, judge_fallback = run_judge_only(
            self.ollama,
            cfg=cfg,
            goal=goal,
            constraints=constraints,
            candidates=enriched_candidates,
            baseline_actions=baseline_actions,
            latest_advocate=latest_advocate,
            latest_critic=latest_critic,
            human_feedback=human_feedback,
            run_dir=run_dir,
        )

        ranked_actions = self._merge_judge_output(
            final_actions, enriched_by_pid, horizon_days, constraints=constraints
        )
        judge_used = bool(judge_raw.get("judge_used", not judge_fallback))
        judge_pid_warning = judge_raw.get("judge_pid_warning")
        self._write_stage(run_dir, "9_final.json", {
            "ranked_actions": ranked_actions,
            "judge_used": judge_used,
            "judge_pid_warning": judge_pid_warning,
        })

        return {
            "ok": True,
            "ranked_actions": ranked_actions,
            "judge_raw": judge_raw,
            "judge_fallback": judge_fallback,
            "judge_used": judge_used,
            "judge_pid_warning": judge_pid_warning,
        }
