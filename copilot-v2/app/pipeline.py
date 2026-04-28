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
    ) -> None:
        self.snapshot_id = snapshot_id
        root = artifacts_root or Path(__file__).resolve().parent.parent / "artifacts"

        self.retrieval = RetrievalAgent(snapshot_id=snapshot_id, artifacts_root=root)
        min_score_env = os.environ.get("COPILOT_RETRIEVAL_MIN_SCORE", "").strip()
        if min_score_env:
            try:
                self.retrieval.config.min_score = float(min_score_env)
            except Exception:
                pass
        self.retrieval.load_index()

        self.pricing = PricingAgent(snapshot_id=snapshot_id, artifacts_root=root)
        self.sentiment = SentimentAgent(snapshot_id=snapshot_id, artifacts_root=root)
        self.inventory = InventoryAgent(snapshot_id=snapshot_id, artifacts_root=root)

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
    ) -> dict[str, Any]:
        """Preview retrieval results (fast) so UI can show match count and examples."""
        constraints = constraints or {}
        # Use the same rewrite behavior as pipeline.
        rewrite = self._rewrite_goal_for_retrieval(goal)
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

        def _field(doc: str, key: str) -> str:
            m = re.search(rf"^{re.escape(key)}:\\s*(.*)$", doc, flags=re.M)
            return (m.group(1).strip() if m else "")

        cands = []
        for r in results:
            doc = str(r.get("product_document") or "")
            cands.append({
                "product_id": str(r.get("product_id") or ""),
                "score": float(r.get("retrieval_score") or 0.0),
                "title": _field(doc, "title")[:140],
                "category": _field(doc, "category")[:80],
                "subcategory": _field(doc, "subcategory")[:80],
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

    def _rewrite_goal_for_retrieval(self, goal: str) -> dict[str, Any]:
        """Rewrite a broad business goal into a product-oriented retrieval query.

        This is intentionally lightweight and safe:
        - If rewriting is disabled or fails, we fall back to the original goal.
        - Output is persisted to runs as `0_query_rewrite.json` for debugging.
        """
        goal = str(goal or "").strip()
        original_goal = goal
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
        goal_for_logic = goal_fixed

        # Heuristic: only rewrite when the goal looks broad/strategy-level.
        broad_markers = [
            "increase revenue", "increase profit", "grow revenue", "grow sales", "boost sales",
            "improve business", "improve performance", "increase conversion", "increase margin",
            "reduce returns", "reduce churn", "reduce complaints",
        ]
        goal_l = goal_for_logic.lower()
        # Treat goals like "grow revenue for Baking Cups ..." as already specific enough.
        has_specific_for_phrase = bool(
            re.search(r"\bfor\s+[a-z0-9][a-z0-9&'\\-]{3,}(?:\s+[a-z0-9][a-z0-9&'\\-]{2,}){0,4}\b", goal_l)
        )
        if has_specific_for_phrase:
            # Avoid rewriting if user already provided a concrete product/category after "for".
            if typo_fixes and goal_fixed != original_goal:
                note = "; ".join([f"{f['from']} → {f['to']}" for f in typo_fixes])
                return {
                    "ok": True,
                    "used": True,
                    "original_goal": original_goal,
                    "retrieval_query": goal_fixed,
                    "notes": f"Typo corrected: {note}",
                }
            return {"ok": True, "used": False, "original_goal": original_goal, "retrieval_query": original_goal}

        is_broad = any(m in goal_l for m in broad_markers) and len(goal_for_logic.split()) <= 12
        if not is_broad:
            if typo_fixes and goal_fixed != original_goal:
                note = "; ".join([f"{f['from']} → {f['to']}" for f in typo_fixes])
                return {
                    "ok": True,
                    "used": True,
                    "original_goal": original_goal,
                    "retrieval_query": goal_fixed,
                    "notes": f"Typo corrected: {note}",
                }
            return {"ok": True, "used": False, "original_goal": original_goal, "retrieval_query": original_goal}

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
        if inv in {"low_stock"} or risk_flag:
            return "hold"

        # Missing pricing coverage: don't pretend we can reprice intelligently.
        if src == "fallback":
            return "investigate"

        # Objective-specific nudges
        if obj == "avoid_stockouts":
            return "hold"
        if obj == "reduce_returns" and (total_returns >= 2 or (n_rev >= 8 and p_neg >= 0.20)):
            return "investigate"

        # Moderate-quality issues should surface as investigate (not only extreme cases)
        if total_returns >= 3:
            return "investigate"
        if n_rev >= 10 and p_neg >= 0.30:
            return "investigate"

        # Overstock: prefer promote unless quality signals suggest investigating first.
        if inv == "overstocked":
            return "promote"

        # For clear-inventory objective, promote/reprice down is often safer than raising price.
        if obj == "clear_inventory" and pchg > 0.0:
            return "promote"

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
            price_change_raw = (
                float(pricing_result.get("predicted_price_change_pct") or 0.0)
                if pricing_result.get("found")
                else 0.0
            )
            pricing_source = "cache" if pricing_result.get("found") else "fallback"
            # Flags for peaky outputs / big deltas.
            large_delta = bool(pricing_result.get("found")) and abs(price_change_raw) >= 0.7 * policy_bound
            near_bound = bool(pricing_result.get("found")) and abs(price_change_raw) >= 0.9 * policy_bound
            pricing_info: dict[str, Any] = {
                "source": pricing_source,
                # Proxy for price missing / pricing unavailable in this snapshot.
                # (If pricing cache has no entry, we cannot ground price-based rationale.)
                "price_missing": (pricing_source != "cache"),
                "policy_bound": policy_bound,
                "large_delta": large_delta,
                "near_bound": near_bound,
                "predicted_price_change_pct_raw": price_change_raw if pricing_result.get("found") else None,
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
            returns = float(inv.get("total_returns") or 0.0)

            inventory_info = {
                "stock_status": inv.get("stock_status", "unknown"),
                "risk_flag": bool(inv.get("risk_flag", False)),
                "signals": {
                    "on_hand_units": on_hand,
                    "safety_stock_units": safety,
                    "available_to_sell": available,
                    "mean_daily_revenue": mean_rev,
                    "total_returns": returns,
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
                "total_returns": returns,
                "on_hand_units": on_hand,
                "safety_stock_units": safety,
            }

            # Optional shrinkage for large deltas (post soft-cap). Keep behind a flag for A/B.
            price_change = float(price_change_raw or 0.0)
            if enable_shrink and pricing_source == "cache" and large_delta:
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

            suggested_action = self._suggest_action(
                objective=str((constraints or {}).get("objective") or "revenue"),
                inventory_status=str(inventory_info.get("stock_status") or "unknown"),
                risk_flag=bool(inventory_info.get("risk_flag", False)),
                pricing_source=str(pricing_info.get("source") or "fallback"),
                predicted_price_change_pct=price_change,
                n_reviews=float(sentiment_info.get("n_reviews") or 0.0),
                p_neg=p_neg,
                total_returns=returns,
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
                "action_type": "reprice",
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

    def _make_baseline(
        self, enriched: list[dict[str, Any]], top_n: int
    ) -> list[dict[str, Any]]:
        sorted_cands = sorted(
            enriched, key=lambda c: c["evidence"]["retrieval_score"], reverse=True
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
            if returns > 0:
                out.append(f"Returns total_returns={int(returns)}.")

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
            # If LLM rationale is missing or looks generic, replace with deterministic bullets.
            if _looks_generic(llm_rationale):
                llm_rationale = det
            out.append({
                **base,
                "product_id": pid,
                "action_type": str(ja.get("action_type") or "reprice"),
                "recommended_price_change_pct": float(ja.get("recommended_price_change_pct") or 0.0),
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
    ) -> dict[str, Any]:
        """Fast path: retrieval + enrichment + baseline only. No LLM calls."""
        constraints = constraints or {"max_abs_price_change_pct": 10.0}
        ts = datetime.now(timezone.utc)
        owner_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", owner_id)[:16]
        run_id = f"{ts.strftime('%Y%m%d_%H%M%S')}_{owner_slug}"
        run_dir = self._runs_root / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        rewrite = self._rewrite_goal_for_retrieval(goal)
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
        baseline_ranked = self._make_baseline(baseline_for_judge, top_n_actions)
        self._write_stage(run_dir, "3_baseline.json", baseline_ranked)

        trace = {
            "snapshot_id": self.snapshot_id,
            "owner_id": owner_id,
            "retrieval_index_meta": self._index_meta,
            "query_rewrite": {k: rewrite.get(k) for k in ("used", "retrieval_query", "clarifying_question", "notes")},
            "pricing": {"wired": enable_pricing, "source": "cache"},
            "sentiment": {"wired": enable_sentiment, "source": "cache"},
            "inventory": {"wired": True, "mode": "rules_v1"},
            "demo_allowlist_size": 0,
        }

        return {
            "ok": True,
            "goal": goal,
            "enriched_candidates": enriched,
            "baseline_ranked_actions": baseline_ranked,
            "trace": trace,
        }

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
    ) -> dict[str, Any]:
        """Advocate + Critic round 1 on pre-enriched candidates. No retrieval."""
        constraints = constraints or {"max_abs_price_change_pct": 10.0}
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
        }
        self._write_stage(run_dir, "0_config.json", config)

        # 1. Retrieve candidates
        rewrite = self._rewrite_goal_for_retrieval(goal)
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
        baseline_ranked = self._make_baseline(baseline_for_judge, top_n_actions)
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
            "pricing": {"wired": enable_pricing, "source": "cache"},
            "sentiment": {"wired": enable_sentiment, "source": "cache"},
            "inventory": {"wired": True, "mode": "rules_v1"},
            "demo_allowlist_size": 0,
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

        return response

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
    ) -> dict[str, Any]:
        """Run the judge exactly once on the final advocate + critic outputs."""
        constraints = constraints or {"max_abs_price_change_pct": 10.0}
        enriched_by_pid = {c["product_id"]: c for c in enriched_candidates}
        horizon_days = int(enriched_candidates[0].get("horizon_days", 7)) if enriched_candidates else 7

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

        ranked_actions = self._merge_judge_output(final_actions, enriched_by_pid, horizon_days)
        self._write_stage(run_dir, "9_final.json", {"ranked_actions": ranked_actions})

        return {
            "ok": True,
            "ranked_actions": ranked_actions,
            "judge_raw": judge_raw,
            "judge_fallback": judge_fallback,
        }
