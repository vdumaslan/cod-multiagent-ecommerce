"""Pydantic request/response models for the React client."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Constraints(BaseModel):
    max_abs_price_change_pct: float = 10.0


class OrchestrateRequest(BaseModel):
    goal: str
    owner_id: str
    horizon_days: int = 7
    top_n_actions: int = 3
    constraints: Constraints = Field(default_factory=Constraints)
    use_llm_policy: bool = True
    enable_pricing: bool = True
    enable_sentiment: bool = True
    advocate_model: str = "llama3.1:8b"
    critic_model: str = "qwen2.5:7b-instruct"
    judge_model: str = "qwen2.5:7b-instruct"
    prompt_style: str = "few_shot_json"
    prompt_version: str = "v1"
    human_review_mode: Literal["skip", "second_round", "second_round_with_feedback"] = "skip"
    human_feedback: str | None = None


class PipelineRequest(BaseModel):
    goal: str
    owner_id: str
    horizon_days: int = 7
    top_n_actions: int = 3
    constraints: Constraints = Field(default_factory=Constraints)
    enable_pricing: bool = True
    enable_sentiment: bool = True


class DebateStartRequest(BaseModel):
    goal: str
    owner_id: str
    constraints: Constraints = Field(default_factory=Constraints)
    enriched_candidates: list[dict[str, Any]]
    baseline_actions: list[dict[str, Any]]
    top_n_actions: int = 3
    advocate_model: str = "llama3.1:8b"
    critic_model: str = "qwen2.5:7b-instruct"
    judge_model: str = "qwen2.5:7b-instruct"
    prompt_style: str = "few_shot_json"
    prompt_version: str = "v1"


class EvidencePoint(BaseModel):
    type: str
    text: str
    score: float


class Evidence(BaseModel):
    retrieval_score: float
    points: list[EvidencePoint] = []


class InventorySignals(BaseModel):
    on_hand_units: float
    safety_stock_units: float
    available_to_sell: float
    mean_daily_revenue: float
    total_returns: float


class InventoryResult(BaseModel):
    stock_status: str
    risk_flag: bool
    signals: InventorySignals
    rules_fired: list[str] = []


class ActionSignals(BaseModel):
    margin_pct: float = 0.0
    available_to_sell: float = 0.0
    mean_daily_revenue: float = 0.0
    total_returns: float = 0.0
    on_hand_units: float = 0.0
    safety_stock_units: float = 0.0


class RankedAction(BaseModel):
    product_id: str
    action_type: str
    recommended_price_change_pct: float
    pricing: dict[str, Any]
    sentiment: dict[str, Any]
    signals: ActionSignals
    evidence: Evidence
    horizon_days: int
    inventory: dict[str, Any]
    rank: int = 0
    llm_rationale_bullets: list[str] = []
    llm_risk_bullets: list[str] = []


class TraceInfo(BaseModel):
    snapshot_id: str
    owner_id: str
    retrieval_index_meta: str = ""
    pricing: dict[str, Any] = Field(default_factory=dict)
    sentiment: dict[str, Any] = Field(default_factory=dict)
    inventory: dict[str, Any] = Field(default_factory=dict)
    demo_allowlist_size: int = 0


class PipelineResponse(BaseModel):
    ok: bool
    goal: str
    enriched_candidates: list[dict[str, Any]]
    baseline_ranked_actions: list[RankedAction]
    trace: TraceInfo


class DebateStartResponse(BaseModel):
    ok: bool
    advocate: dict[str, Any]
    critic: dict[str, Any]
    debate_trace: dict[str, Any] | None = None


class OrchestrateResponse(BaseModel):
    ok: bool
    goal: str
    served_from_warm_process: bool = True
    input: dict[str, Any]
    ranked_actions: list[RankedAction]
    baseline_ranked_actions: list[RankedAction]
    enriched_candidates: list[dict[str, Any]] = []
    trace: TraceInfo
    debate_trace: dict[str, Any] | None = None


class DebateContinueRequest(BaseModel):
    goal: str
    owner_id: str
    constraints: Constraints = Field(default_factory=Constraints)
    enriched_candidates: list[dict[str, Any]]
    baseline_actions: list[dict[str, Any]]
    prev_advocate: dict[str, Any]
    prev_critic: dict[str, Any]
    human_feedback: str | None = None
    top_n_actions: int = 3
    advocate_model: str = "llama3.1:8b"
    critic_model: str = "qwen2.5:7b-instruct"
    judge_model: str = "qwen2.5:7b-instruct"
    prompt_style: str = "few_shot_json"
    prompt_version: str = "v1"


class DebateContinueResponse(BaseModel):
    ok: bool
    advocate: dict[str, Any]
    critic: dict[str, Any]
    raw: dict[str, Any] = Field(default_factory=dict)


class DebateJudgeRequest(BaseModel):
    goal: str
    owner_id: str
    constraints: Constraints = Field(default_factory=Constraints)
    enriched_candidates: list[dict[str, Any]]
    baseline_actions: list[dict[str, Any]]
    latest_advocate: dict[str, Any]
    latest_critic: dict[str, Any]
    human_feedback: str | None = None
    top_n_actions: int = 3
    judge_model: str = "qwen2.5:7b-instruct"
    prompt_style: str = "few_shot_json"
    prompt_version: str = "v1"


class DebateJudgeResponse(BaseModel):
    ok: bool
    ranked_actions: list[RankedAction]
    judge_raw: dict[str, Any] = Field(default_factory=dict)
    judge_fallback: bool = False


class HealthResponse(BaseModel):
    ok: bool
    snapshot_id: str
    has_pricing_cache: bool
    has_sentiment_cache: bool
    has_inventory_cache: bool
