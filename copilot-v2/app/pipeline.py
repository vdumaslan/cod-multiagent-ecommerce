"""Main entry point: wire retrieval → specialist enrichment → optional ACJ debate."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agents.retrieval_agent import RetrievalAgent
from app.agents.pricing_agent import PricingAgent
from app.agents.sentiment_agent import SentimentAgent
from app.agents.inventory_agent import InventoryAgent
from app.agents.orchestrator.orchestrator import ACJConfig, run_acj, continue_acj, run_judge_only
from app.llm import OllamaClient

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
        self.retrieval.load_index()

        self.pricing = PricingAgent(snapshot_id=snapshot_id, artifacts_root=root)
        self.sentiment = SentimentAgent(snapshot_id=snapshot_id, artifacts_root=root)
        self.inventory = InventoryAgent(snapshot_id=snapshot_id, artifacts_root=root)

        self.ollama = OllamaClient(base_url=ollama_base_url)
        self._index_meta = str(
            root / "indexes" / snapshot_id / "dense" / "intfloat_e5-large-v2" / "index_meta.json"
        )
        self._runs_root = root / "runs" / snapshot_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enrich(
        self,
        candidates: list[dict[str, Any]],
        *,
        horizon_days: int,
        enable_pricing: bool,
        enable_sentiment: bool,
    ) -> list[dict[str, Any]]:
        enriched = []
        for c in candidates:
            pid = str(c.get("product_id") or "")
            if not pid:
                continue

            # Pricing
            pricing_result = self.pricing.lookup(pid) if enable_pricing else {"found": False}
            price_change = (
                float(pricing_result.get("predicted_price_change_pct") or 0.0)
                if pricing_result.get("found")
                else 0.0
            )
            pricing_info = {
                "source": "cache" if pricing_result.get("found") else "fallback"
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

            enriched.append({
                "product_id": pid,
                "action_type": "reprice",
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
        out = []
        for i, ja in enumerate(judge_actions):
            pid = str(ja.get("product_id") or "")
            base = enriched_by_pid.get(pid, {})
            out.append({
                **base,
                "product_id": pid,
                "action_type": str(ja.get("action_type") or "reprice"),
                "recommended_price_change_pct": float(ja.get("recommended_price_change_pct") or 0.0),
                "horizon_days": horizon_days,
                "rank": i + 1,
                "llm_rationale_bullets": list(ja.get("rationale_bullets") or []),
                "llm_risk_bullets": list(ja.get("risk_bullets") or []),
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

        candidates_raw = self.retrieval.retrieve(goal)
        self._write_stage(run_dir, "1_retrieval.json", candidates_raw)

        enriched = self._enrich(
            candidates_raw,
            horizon_days=horizon_days,
            enable_pricing=enable_pricing,
            enable_sentiment=enable_sentiment,
        )
        self._write_stage(run_dir, "2_enriched.json", enriched)

        baseline_for_judge = [{**c, "recommended_price_change_pct": 0.0} for c in enriched]
        baseline_ranked = self._make_baseline(baseline_for_judge, top_n_actions)
        self._write_stage(run_dir, "3_baseline.json", baseline_ranked)

        trace = {
            "snapshot_id": self.snapshot_id,
            "owner_id": owner_id,
            "retrieval_index_meta": self._index_meta,
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
        candidates_raw = self.retrieval.retrieve(goal)
        self._write_stage(run_dir, "1_retrieval.json", candidates_raw)

        # 2. Enrich with specialist signals
        enriched = self._enrich(
            candidates_raw,
            horizon_days=horizon_days,
            enable_pricing=enable_pricing,
            enable_sentiment=enable_sentiment,
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
