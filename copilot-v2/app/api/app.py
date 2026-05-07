"""ASGI application factory and route registration."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.schemas import (
    HealthResponse,
    CatalogSummaryResponse,
    CatalogFacetsResponse,
    RetrievalPreviewRequest,
    RetrievalPreviewResponse,
    OrchestrateRequest,
    OrchestrateResponse,
    PipelineRequest,
    PipelineResponse,
    DebateStartRequest,
    DebateStartResponse,
    DebateContinueRequest,
    DebateContinueResponse,
    DebateJudgeRequest,
    DebateJudgeResponse,
    ABEventRequest,
    ABSaveRunRequest,
)
from app.ab import assign_variant, log_event, save_version_a_run
from app.rl import default_arms, load_state, save_state, summarize, update_from_event
from app.pipeline import Pipeline, SNAPSHOT_ID

app = FastAPI(title="Seller Copilot API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        snapshot_id = os.environ.get("COPILOT_SNAPSHOT_ID", SNAPSHOT_ID)
        artifacts_root_env = os.environ.get("COPILOT_ARTIFACTS_ROOT", "")
        artifacts_root = Path(artifacts_root_env) if artifacts_root_env else None
        ollama_url = os.environ.get("COPILOT_OLLAMA_URL", "http://localhost:11434")
        data_backend = os.environ.get("COPILOT_DATA_BACKEND", "local")
        _pipeline = Pipeline(
            snapshot_id=snapshot_id,
            artifacts_root=artifacts_root,
            ollama_base_url=ollama_url,
            data_backend=data_backend,
        )
    return _pipeline


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    p = get_pipeline()
    return HealthResponse(
        ok=True,
        snapshot_id=p.snapshot_id,
        has_pricing_cache=bool(p.pricing._cache),
        has_sentiment_cache=bool(p.sentiment._cache),
        has_inventory_cache=bool(p.inventory._cache),
        has_retrieval_index=p.retrieval._index is not None,
    )


@app.get("/catalog/summary", response_model=CatalogSummaryResponse)
def catalog_summary() -> CatalogSummaryResponse:
    p = get_pipeline()
    return CatalogSummaryResponse(ok=True, **p.catalog_summary())


@app.get("/catalog/facets", response_model=CatalogFacetsResponse)
def catalog_facets() -> CatalogFacetsResponse:
    p = get_pipeline()
    return CatalogFacetsResponse(ok=True, **p.catalog_facets())


@app.post("/retrieval/preview", response_model=RetrievalPreviewResponse)
def retrieval_preview(req: RetrievalPreviewRequest) -> RetrievalPreviewResponse:
    p = get_pipeline()
    out = p.retrieval_preview(goal=req.goal, constraints=req.constraints.model_dump(), top_k_preview=req.top_k_preview)
    return RetrievalPreviewResponse(**out)


@app.post("/pipeline", response_model=PipelineResponse)
def pipeline(req: PipelineRequest) -> PipelineResponse:
    if not req.owner_id:
        raise HTTPException(status_code=400, detail="missing_owner_id")
    p = get_pipeline()
    result = p.run_pipeline(
        goal=req.goal,
        owner_id=req.owner_id,
        horizon_days=req.horizon_days,
        top_n_actions=req.top_n_actions,
        constraints=req.constraints.model_dump(),
        enable_pricing=req.enable_pricing,
        enable_sentiment=req.enable_sentiment,
    )
    return PipelineResponse(**result)


@app.post("/debate/start", response_model=DebateStartResponse)
def debate_start(req: DebateStartRequest) -> DebateStartResponse:
    if not req.owner_id:
        raise HTTPException(status_code=400, detail="missing_owner_id")
    p = get_pipeline()
    result = p.start_debate(
        goal=req.goal,
        owner_id=req.owner_id,
        constraints=req.constraints.model_dump(),
        enriched_candidates=req.enriched_candidates,
        baseline_actions=req.baseline_actions,
        top_n_actions=req.top_n_actions,
        advocate_model=req.advocate_model,
        critic_model=req.critic_model,
        judge_model=req.judge_model,
        prompt_style=req.prompt_style,
        prompt_version=req.prompt_version,
        run_id=req.run_id,
    )
    return DebateStartResponse(**result)


@app.post("/debate/continue", response_model=DebateContinueResponse)
def debate_continue(req: DebateContinueRequest) -> DebateContinueResponse:
    if not req.owner_id:
        raise HTTPException(status_code=400, detail="missing_owner_id")
    p = get_pipeline()
    result = p.continue_debate(
        goal=req.goal,
        owner_id=req.owner_id,
        constraints=req.constraints.model_dump(),
        enriched_candidates=req.enriched_candidates,
        baseline_actions=req.baseline_actions,
        prev_advocate=req.prev_advocate,
        prev_critic=req.prev_critic,
        top_n_actions=req.top_n_actions,
        advocate_model=req.advocate_model,
        critic_model=req.critic_model,
        judge_model=req.judge_model,
        prompt_style=req.prompt_style,
        prompt_version=req.prompt_version,
        human_feedback=req.human_feedback,
    )
    return DebateContinueResponse(**result)


@app.post("/debate/judge", response_model=DebateJudgeResponse)
def debate_judge(req: DebateJudgeRequest) -> DebateJudgeResponse:
    if not req.owner_id:
        raise HTTPException(status_code=400, detail="missing_owner_id")
    p = get_pipeline()
    result = p.run_judge(
        goal=req.goal,
        owner_id=req.owner_id,
        constraints=req.constraints.model_dump(),
        enriched_candidates=req.enriched_candidates,
        baseline_actions=req.baseline_actions,
        latest_advocate=req.latest_advocate,
        latest_critic=req.latest_critic,
        top_n_actions=req.top_n_actions,
        judge_model=req.judge_model,
        prompt_style=req.prompt_style,
        prompt_version=req.prompt_version,
        human_feedback=req.human_feedback,
        run_id=req.run_id,
    )
    return DebateJudgeResponse(**result)


@app.get("/ab/variant/{owner_id}")
def ab_variant(owner_id: str) -> dict:
    return {"owner_id": owner_id, "variant": assign_variant(owner_id)}


@app.post("/ab/event")
def ab_event(req: ABEventRequest) -> dict:
    log_event(
        owner_id=req.owner_id,
        variant=req.variant,
        run_id=req.run_id,
        event=req.event,
        metadata=req.metadata,
    )
    # RL updates are applied only for Version B runs.
    try:
        mode_used = str((req.metadata or {}).get("mode_used") or "").strip().upper()
        if mode_used == "B" and req.run_id:
            arms = default_arms()
            st = load_state(arms)
            info = update_from_event(st, run_id=req.run_id, event=req.event, metadata=req.metadata)
            save_state(st)
            return {"ok": True, "rl_update": info}
    except Exception:
        pass
    return {"ok": True}


@app.get("/rl/stats")
def rl_stats() -> dict:
    """Return current bandit arm stats (Version B only)."""
    arms = default_arms()
    st = load_state(arms)
    return summarize(st, arms)


@app.post("/ab/save_run")
def ab_save_run(req: ABSaveRunRequest) -> dict:
    p = get_pipeline()
    run_id = save_version_a_run(
        owner_id=req.owner_id,
        variant=req.variant,
        goal=req.goal,
        retrieval_candidates=req.retrieval_candidates,
        decisions=req.decisions,
        time_to_decision_s=req.time_to_decision_s,
        confidence_rating=req.confidence_rating,
        runs_root=p._runs_root,
    )
    return {"ok": True, "run_id": run_id}


@app.post("/orchestrate", response_model=OrchestrateResponse)
def orchestrate(req: OrchestrateRequest) -> OrchestrateResponse:
    if not req.owner_id:
        raise HTTPException(status_code=400, detail="missing_owner_id")
    p = get_pipeline()
    result = p.run(
        goal=req.goal,
        owner_id=req.owner_id,
        horizon_days=req.horizon_days,
        top_n_actions=req.top_n_actions,
        constraints=req.constraints.model_dump(),
        use_llm_policy=req.use_llm_policy,
        enable_pricing=req.enable_pricing,
        enable_sentiment=req.enable_sentiment,
        advocate_model=req.advocate_model,
        critic_model=req.critic_model,
        judge_model=req.judge_model,
        prompt_style=req.prompt_style,
        prompt_version=req.prompt_version,
        human_review_mode=req.human_review_mode,
        human_feedback=req.human_feedback,
    )
    return OrchestrateResponse(**result)
