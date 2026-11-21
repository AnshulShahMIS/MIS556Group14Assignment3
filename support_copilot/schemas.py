from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Output(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Triage(Output):
    intent: Literal['refund', 'delay', 'account', 'bug', 'praise', 'other']
    sentiment: float = Field(ge=-1, le=1)
    urgency: Literal['low', 'med', 'high']
    risk_flags: list[str]
    needs_human: bool
    off_topic: bool
    spam: bool = False
    reason: str


class Draft(Output):
    reply: str
    policy_ids: list[str]
    case_ids: list[str]
    uncertainty: str


class PolicyAssessment(Output):
    relevant_policy_ids: list[str]
    evidence_sufficient: bool
    requires_human: bool
    reason: str


class Critique(Output):
    passed: bool
    issues: list[str]
    supported: bool
    needs_human: bool
    reason: str


class ScoreEntry(Output):
    variant: Literal['A','B','C']
    helpfulness: int = Field(ge=1,le=5)
    factual_support: int = Field(ge=1,le=5)
    tone: int = Field(ge=1,le=5)
    appropriate_escalation: bool


class Scores(Output):
    scores: list[ScoreEntry]
    rationale: str
