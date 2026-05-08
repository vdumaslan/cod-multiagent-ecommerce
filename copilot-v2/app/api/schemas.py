"""Pydantic request/response models for the React client."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Constraints(BaseModel):
    max_abs_price_change_pct: float = 10.0
    # Accept a few aliases to keep older UIs / test harnesses working.
    objective: Literal[
        "revenue",
        "grow_revenue",
        "profit",
        "profit_proxy",
        "clear_inventory",
        "clear-inventory",
        "liquidation",
        "reduce_returns",
        "avoid_stockouts",
    ] = "revenue"
    do_not_raise_if_p_neg_above: float | None = None
    exclude_low_stock: bool = False
    exclude_stockout_risk: bool = False

    @field_validator("objective", mode="before")
    @classmethod
    def _normalize_objective(cls, v: Any) -> Any:
        if v is None:
            return v
        s = str(v).strip().lower()
        if s in {"profit_proxy"}:
            return "profit"
        if s in {"clear-inventory"}:
            return "clear_inventory"
        if s in {"liquidation"}:
            return "clear_inventory"
        if s in {"grow revenue", "grow_rev"}:
            return "grow_revenue"
        return s


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
    selected_category: str = ""
    selected_subcategory: str = ""


class PipelineRequest(BaseModel):
    goal: str
    owner_id: str
    horizon_days: int = 7
    top_n_actions: int = 3
    constraints: Constraints = Field(default_factory=Constraints)
    enable_pricing: bool = True
    enable_sentiment: bool = True
    # UI scope selection — used to derive a clean retrieval query instead of
    # feeding the full business-goal sentence to FAISS.
    selected_category: str = ""
    selected_subcategory: str = ""


class DebateStartRequest(BaseModel):
    goal: str
    owner_id: str
    run_id: str | None = None
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
    total_units_sold: float = 0.0
    total_returns: float
    return_rate: float | None = None


class InventoryResult(BaseModel):
    stock_status: str
    risk_flag: bool
    signals: InventorySignals
    rules_fired: list[str] = []


class ActionSignals(BaseModel):
    margin_pct: float = 0.0
    available_to_sell: float = 0.0
    mean_daily_revenue: float = 0.0
    total_units_sold: float = 0.0
    total_returns: float = 0.0
    return_rate: float | None = None
    on_hand_units: float = 0.0
    safety_stock_units: float = 0.0


class RankedAction(BaseModel):
    product_id: str
    action_type: str
    suggested_action: str = ""
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
    query_rewrite: dict[str, Any] = Field(default_factory=dict)
    pricing: dict[str, Any] = Field(default_factory=dict)
    sentiment: dict[str, Any] = Field(default_factory=dict)
    inventory: dict[str, Any] = Field(default_factory=dict)
    demo_allowlist_size: int = 0
    rl: dict[str, Any] = Field(default_factory=dict)


class PipelineResponse(BaseModel):
    ok: bool
    run_id: str = ""
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
    run_id: str | None = None
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
    judge_used: bool | None = None
    judge_pid_warning: str | None = None


class ABEventRequest(BaseModel):
    owner_id: str
    variant: str
    run_id: str = ""
    event: Literal["session_start", "decision_made", "abandoned", "confidence_rated"]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ABSaveRunRequest(BaseModel):
    owner_id: str
    variant: str
    goal: str
    retrieval_candidates: list[dict[str, Any]] = Field(default_factory=list)
    decisions: dict[str, Any] = Field(default_factory=dict)
    time_to_decision_s: int | None = None
    confidence_rating: int | None = None


class HealthResponse(BaseModel):
    ok: bool
    snapshot_id: str
    has_pricing_cache: bool
    has_sentiment_cache: bool
    has_inventory_cache: bool
    has_retrieval_index: bool


class CatalogBucket(BaseModel):
    name: str
    count: int


class CatalogSummaryResponse(BaseModel):
    ok: bool
    snapshot_id: str
    n_products: int
    top_categories: list[CatalogBucket] = Field(default_factory=list)
    top_subcategories: list[CatalogBucket] = Field(default_factory=list)


class CatalogFacetsResponse(BaseModel):
    ok: bool
    snapshot_id: str
    n_products: int
    categories: list[CatalogBucket] = Field(default_factory=list)
    subcategories_by_category: dict[str, list[CatalogBucket]] = Field(default_factory=dict)


class RetrievalPreviewCandidate(BaseModel):
    product_id: str
    score: float
    title: str = ""
    category: str = ""
    subcategory: str = ""


class RetrievalPreviewRequest(BaseModel):
    goal: str
    constraints: Constraints = Field(default_factory=Constraints)
    top_k_preview: int = 5
    selected_category: str = ""
    selected_subcategory: str = ""


class RetrievalPreviewResponse(BaseModel):
    ok: bool
    goal: str
    retrieval_query: str = ""
    clarifying_question: str | None = None
    min_score: float = 0.0
    n_candidates_above_min_score: int = 0
    top_candidates: list[RetrievalPreviewCandidate] = Field(default_factory=list)
