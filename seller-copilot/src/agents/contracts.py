from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentOutput(BaseModel):
    agent_name: str
    claim: str
    recommended_items: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    risks_or_limitations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DebateResult(BaseModel):
    winner: str
    runner_up: str | None = None
    rationale: str
    uncertainty: float = Field(ge=0.0, le=1.0)
    traces: list[AgentOutput]

