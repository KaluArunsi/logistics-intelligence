"""
Backend API contracts for runtime orchestration.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SessionStatusResponse(BaseModel):
    has_session: bool
    session_expires_at: Optional[str] = None
    cooldown_until: Optional[str] = None
    state: Optional[str] = None
    last_successful_step_uri: Optional[str] = None
    active_task_id: Optional[str] = None
    report_id: Optional[str] = None
    artifacts: dict[str, Optional[str]] = Field(default_factory=dict)


class TellUsRequest(BaseModel):
    problem_statement: str
    industry: Optional[str] = None
    context: dict[str, Any] = Field(default_factory=dict)


class TellUsConfirmRequest(BaseModel):
    industry: Optional[str] = None
    use_synthetic: bool = True


class TellUsQuestionsRequest(BaseModel):
    industry: Optional[str] = None
    category: Optional[str] = None


class TellUsAugmentRequest(BaseModel):
    missing_fields: list[str] = Field(default_factory=list)
    provided_values: dict[str, Any] = Field(default_factory=dict)


class TellUsTurnRequest(BaseModel):
    industry: Optional[str] = None
    category: Optional[str] = None
    answer: Optional[str] = None
    missing_fields: list[str] = Field(default_factory=list)
    max_questions: int = Field(default=10, ge=1, le=25)
    reset: bool = False


class TellUsTurnResponse(BaseModel):
    state: str
    industry: str
    category: str
    completed: bool
    question_index: int
    total_questions: int
    question: Optional[dict[str, Any]] = None
    parse_confidence: Optional[float] = None
    answer_source: Optional[str] = None
    normalized_answer: Optional[Any] = None
    assistant_message: Optional[str] = None
    conversation_uri: Optional[str] = None
    question_source: Optional[str] = None
    question_mode: Optional[str] = None
    question_provider: Optional[str] = None
    question_model: Optional[str] = None
    summary: dict[str, int] = Field(default_factory=dict)


class TellUsFinalizeRequest(BaseModel):
    industry: Optional[str] = None
    category: Optional[str] = None
    force: bool = False


class TellUsFinalizeResponse(BaseModel):
    state: str
    industry: str
    category: str
    tell_us_uri: str
    conversation_uri: Optional[str] = None
    summary: dict[str, int] = Field(default_factory=dict)
    next_step: str = "/intake/tell-us/confirm"


class SchemaAnalyzeRequest(BaseModel):
    artifact_uri: Optional[str] = None


class CategoryMatchRequest(BaseModel):
    industry: Optional[str] = None
    artifact_uri: Optional[str] = None
    columns: list[str] = Field(default_factory=list)


class PredictRequest(BaseModel):
    industry: Optional[str] = None
    category: Optional[str] = None
    artifact_uri: Optional[str] = None
    allow_fallback: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatMessageRequest(BaseModel):
    message: str = Field(..., max_length=4000)
    report_id: Optional[str] = None


class ChatBriefRequest(BaseModel):
    report_id: Optional[str] = None


class ChatExportRequest(BaseModel):
    report_id: Optional[str] = None


class ColumnQuestionsRequest(BaseModel):
    industry: str
    category: str


class ChatGuidedQuestionsRequest(BaseModel):
    report_id: Optional[str] = None
    missing_fields: list[str] = Field(default_factory=list)
    max_questions: int = Field(default=10, ge=1, le=25)


class TaskStatusResponse(BaseModel):
    task_id: str
    state: str
    progress: float
    updated_at: str
    error: Optional[str] = None
    report_id: Optional[str] = None
    industry: Optional[str] = None
    result: Optional[dict] = None
