"""ASGI application factory and route registration."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.schemas import (
    HealthResponse,
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
)
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
        _pipeline = Pipeline(
            snapshot_id=snapshot_id,
            artifacts_root=artifacts_root,
            ollama_base_url=ollama_url,
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
    )


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
    )
    return DebateJudgeResponse(**result)


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
