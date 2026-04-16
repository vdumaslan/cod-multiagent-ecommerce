from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from copilot_v2.llm.ollama_client import OllamaClient
from copilot_v2.runtime.debate import (
    DebateACJConfig,
    DebateConfig,
    run_advocate_critic_judge_safe,
    run_chain_of_debate_safe,
)
from copilot_v2.runtime.cache_io import load_grounding_caches_into_ctx, resolve_grounding_cache_dir
from copilot_v2.runtime.orchestrator import (
    OrchestratorConfig,
    OrchestratorContext,
    _load_inventory,
    _load_sales_summary,
    _load_product_signals,
    _load_products,
    _load_retriever_from_index_meta,
    _read_registry,
    build_ranked_plans,
)

_ACJ_PROMPT_STYLES = frozenset({"zero_shot_json", "few_shot_json", "cot_hidden", "structured_rationale"})


class _State:
    ctx: OrchestratorContext | None = None
    cfg: OrchestratorConfig | None = None
    ollama: OllamaClient | None = None
    owner_retrievers: dict[str, dict[str, Any]] = {}


def _owner_index_meta_path(*, artifacts_root: Path, snapshot_id: str, owner_id: str) -> Path:
    return artifacts_root / "indexes" / snapshot_id / "owners" / owner_id / "dense" / "intfloat_e5-large-v2" / "index_meta.json"


def _get_owner_retriever(*, owner_id: str) -> dict[str, Any] | None:
    """
    Return a retriever scoped to owner_id if owner indexes exist.
    Cached in-process.
    """
    oid = str(owner_id)
    if oid in _State.owner_retrievers:
        return _State.owner_retrievers[oid]
    assert _State.cfg is not None
    p = _owner_index_meta_path(artifacts_root=_State.cfg.artifacts_root, snapshot_id=_State.cfg.snapshot_id, owner_id=oid)
    if not p.is_file():
        return None
    r = _load_retriever_from_index_meta(p)
    _State.owner_retrievers[oid] = r
    return r


def _maybe_load_grounding_cache(ctx: OrchestratorContext, cache_dir: Path) -> None:
    # Backwards-compatible loader; now supports JSON or Parquet.
    load_grounding_caches_into_ctx(ctx=ctx, cache_dir=cache_dir)


class Handler(BaseHTTPRequestHandler):
    server_version = "copilot_v2_orchestrator/1"

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        b = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _read_json(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(n) if n > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json(
                200,
                {
                    "ok": True,
                    "has_ctx": _State.ctx is not None,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "has_pricing_cache": bool(getattr(_State.ctx, "pricing_cache", None)) if _State.ctx else False,
                    "has_sentiment_cache": bool(getattr(_State.ctx, "sentiment_cache", None)) if _State.ctx else False,
                    "demo_allowlist_size": len(getattr(_State.ctx, "demo_allowlist", None) or []) if _State.ctx else 0,
                },
            )
            return
        self._json(404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/orchestrate":
            self._json(404, {"ok": False, "error": "not_found"})
            return
        try:
            body = self._read_json()
            assert _State.cfg is not None and _State.ctx is not None

            goal = str(body.get("goal") or "").strip()
            if not goal:
                self._json(400, {"ok": False, "error": "missing_goal"})
                return

            owner_id = str(body.get("owner_id") or "").strip()
            if not owner_id:
                self._json(400, {"ok": False, "error": "missing_owner_id"})
                return

            top_n_actions = int(body.get("top_n_actions") or _State.cfg.top_n_actions)
            candidate_pool = int(body.get("candidate_pool") or _State.cfg.candidate_pool)
            enable_pricing = bool(body.get("enable_pricing", True))
            enable_sentiment = bool(body.get("enable_sentiment", True))
            horizon_days = int(body.get("horizon_days") or 7)
            constraints = body.get("constraints") if isinstance(body.get("constraints"), dict) else None

            use_llm_policy = bool(body.get("use_llm_policy", False))
            llm_model = str(body.get("llm_model") or "qwen2.5:7b-instruct")
            debate_mode = str(body.get("debate_mode") or "acj").strip().lower()
            advocate_model = str(body.get("advocate_model") or llm_model)
            critic_model = str(body.get("critic_model") or llm_model)
            judge_model = str(body.get("judge_model") or llm_model)
            prompt_style = str(
                body.get("prompt_style")
                or os.environ.get("COPILOT_V2_ACJ_PROMPT_STYLE")
                or "few_shot_json"
            ).strip()
            if prompt_style not in _ACJ_PROMPT_STYLES:
                prompt_style = "few_shot_json"
            prompt_version = str(
                body.get("prompt_version") or os.environ.get("COPILOT_V2_ACJ_PROMPT_VERSION") or "v1"
            ).strip() or "v1"
            debate_top_k = int(body.get("debate_top_k") or min(3, top_n_actions))
            candidate_m = int(body.get("candidate_m") or 20)

            cfg = OrchestratorConfig(
                snapshot_id=_State.cfg.snapshot_id,
                artifacts_root=_State.cfg.artifacts_root,
                registry_path=_State.cfg.registry_path,
                top_n_actions=top_n_actions,
                candidate_pool=candidate_pool,
            )
            owner_retriever = _get_owner_retriever(owner_id=owner_id)
            if owner_retriever is None:
                self._json(400, {"ok": False, "error": "missing_owner_index", "owner_id": owner_id})
                return
            ctx = OrchestratorContext(
                registry=_State.ctx.registry,
                retriever=owner_retriever,
                products=_State.ctx.products,
                signals=_State.ctx.signals,
                inventory=_State.ctx.inventory,
                sales_summary=_State.ctx.sales_summary,
                product_owner=_State.ctx.product_owner,
                pricing_cache=_State.ctx.pricing_cache,
                sentiment_cache=_State.ctx.sentiment_cache,
                demo_allowlist=_State.ctx.demo_allowlist,
            )
            out = build_ranked_plans(
                cfg=cfg,
                goal=goal,
                owner_id=owner_id,
                constraints=constraints,
                horizon_days=horizon_days,
                enable_pricing=enable_pricing,
                enable_sentiment=enable_sentiment,
                ctx=ctx,
            )

            baseline_actions = out.get("ranked_actions") or []
            final_actions = baseline_actions
            debate_trace = None
            if use_llm_policy and _State.ollama is not None and baseline_actions:
                candidates = baseline_actions[: max(3, candidate_m)]
                if debate_mode == "legacy":
                    acts, trace, err = run_chain_of_debate_safe(
                        ollama=_State.ollama,
                        cfg=DebateConfig(model=llm_model, top_k=debate_top_k, candidate_m=candidate_m),
                        goal=goal,
                        constraints=constraints or {},
                        candidates=candidates,
                        baseline_actions=baseline_actions[: debate_top_k],
                    )
                else:
                    acts, trace, err = run_advocate_critic_judge_safe(
                        ollama=_State.ollama,
                        cfg=DebateACJConfig(
                            advocate_model=advocate_model,
                            critic_model=critic_model,
                            judge_model=judge_model,
                            prompt_style=prompt_style,
                            prompt_version=prompt_version,
                            top_k=debate_top_k,
                            candidate_m=candidate_m,
                        ),
                        goal=goal,
                        constraints=constraints or {},
                        candidates=candidates,
                        baseline_actions=baseline_actions[: debate_top_k],
                    )
                debate_trace = trace
                if acts is not None and err is None:
                    # Record debate configuration on success too (not only on error paths).
                    if isinstance(debate_trace, dict):
                        debate_trace["debate_mode"] = debate_mode
                        debate_trace["advocate_model"] = advocate_model
                        debate_trace["critic_model"] = critic_model
                        debate_trace["judge_model"] = judge_model
                        debate_trace["prompt_style"] = prompt_style
                        debate_trace["prompt_version"] = prompt_version

                    by_pid = {str(a.get("product_id")): a for a in baseline_actions}
                    final_actions = []
                    for i, act in enumerate(acts):
                        pid = str(act["product_id"])
                        base = dict(by_pid.get(pid, {"product_id": pid}))
                        base["rank"] = i + 1
                        base["action_type"] = act["action_type"]
                        base["recommended_price_change_pct"] = float(act["recommended_price_change_pct"])
                        base["llm_rationale_bullets"] = act.get("rationale_bullets", [])
                        base["llm_risk_bullets"] = act.get("risk_bullets", [])
                        final_actions.append(base)
                else:
                    # surface candidate IDs for debugging
                    if isinstance(debate_trace, dict):
                        debate_trace["allowed_candidate_ids"] = [str(c.get("product_id")) for c in candidates if c.get("product_id")]
                        debate_trace["debate_mode"] = debate_mode
                        debate_trace["advocate_model"] = advocate_model
                        debate_trace["critic_model"] = critic_model
                        debate_trace["judge_model"] = judge_model
                        debate_trace["prompt_style"] = prompt_style
                        debate_trace["prompt_version"] = prompt_version

            out["baseline_ranked_actions"] = baseline_actions
            out["ranked_actions"] = final_actions
            if debate_trace is not None:
                out["debate_trace"] = debate_trace

            self._json(200, {"ok": True, "goal": goal, "served_from_warm_process": True, **out})
        except Exception as e:  # noqa: BLE001
            self._json(500, {"ok": False, "error": type(e).__name__, "message": str(e)[:500]})


def run_server(
    *,
    snapshot_id: str,
    artifacts_root: Path = Path("copilot-v2/artifacts"),
    registry_path: Path = Path("copilot-v2/artifacts/registry/registry.json"),
    host: str = "127.0.0.1",
    port: int = 8008,
    grounding_cache_dir: Path | None = None,
    grounding_cache_uri: str | None = None,
    demo_allowlist_json: Path | None = None,
) -> None:
    registry = _read_registry(registry_path)
    index_meta = artifacts_root / "indexes" / snapshot_id / "dense" / "intfloat_e5-large-v2" / "index_meta.json"

    cfg = OrchestratorConfig(snapshot_id=snapshot_id, artifacts_root=artifacts_root, registry_path=registry_path)
    ctx = OrchestratorContext(
        registry=registry,
        retriever=_load_retriever_from_index_meta(index_meta),
        products=_load_products(artifacts_root=artifacts_root, snapshot_id=snapshot_id),
        signals=_load_product_signals(artifacts_root=artifacts_root, snapshot_id=snapshot_id),
        inventory=_load_inventory(artifacts_root=artifacts_root, snapshot_id=snapshot_id),
        sales_summary=_load_sales_summary(artifacts_root=artifacts_root, snapshot_id=snapshot_id),
    )
    # Synthetic multi-tenant: deterministically assign each product_id to a store.
    # This enables owner-scoped planning even though the underlying dataset lacks store_id.
    try:
        from copilot_v2.runtime.orchestrator import _default_owner_for_product

        ctx.product_owner = {str(pid): _default_owner_for_product(str(pid)) for pid in ctx.products["product_id"].astype(str).tolist()}
    except Exception:
        ctx.product_owner = None

    resolved_cache_dir, _source = resolve_grounding_cache_dir(
        grounding_cache_dir=grounding_cache_dir,
        grounding_cache_uri=grounding_cache_uri,
    )
    if resolved_cache_dir is not None and resolved_cache_dir.is_dir():
        _maybe_load_grounding_cache(ctx, resolved_cache_dir)
    if demo_allowlist_json is not None and demo_allowlist_json.is_file():
        ids = json.loads(demo_allowlist_json.read_text(encoding="utf-8"))
        if isinstance(ids, list):
            ctx.demo_allowlist = {str(x) for x in ids}

    _State.cfg = cfg
    _State.ctx = ctx
    _State.ollama = OllamaClient(base_url="http://localhost:11434", timeout_s=30.0, retries=1)

    httpd = ThreadingHTTPServer((host, int(port)), Handler)
    print(json.dumps({"ok": True, "host": host, "port": int(port), "snapshot_id": snapshot_id}, indent=2))
    httpd.serve_forever()

