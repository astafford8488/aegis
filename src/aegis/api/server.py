"""AEGIS REST API server — guardrails-as-a-service."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from aegis.guardrails.proxy import GuardrailsProxy
from aegis.guardrails.policy import PolicyEngine
from aegis.detection.classifier import InjectionClassifier
from aegis.agent.sandbox import AgentSandbox, CapabilityBoundary


class ValidateRequest(BaseModel):
    """Input validation request."""
    text: str
    client_id: str = "default"


class ValidateResponse(BaseModel):
    """Input validation response."""
    allowed: bool
    action: str
    risk_level: str
    injection_score: float
    similarity_score: float
    violations: list[dict[str, str]]
    latency_ms: float


class ClassifyRequest(BaseModel):
    """Classification request."""
    text: str


class ClassifyResponse(BaseModel):
    """Classification response."""
    score: float
    label: str
    signals: dict[str, float]
    matched_patterns: list[str]


class ToolCallRequest(BaseModel):
    """Tool call validation request."""
    tool_name: str
    args: dict[str, str]


class ToolCallResponse(BaseModel):
    """Tool call validation response."""
    allowed: bool
    violations: list[dict[str, str]]


def create_app() -> FastAPI:
    """Create the AEGIS API application."""
    app = FastAPI(
        title="AEGIS",
        description="LLM Security Testing & Runtime Guardrails API",
        version="0.1.0",
    )

    proxy = GuardrailsProxy()
    classifier = InjectionClassifier()
    sandbox = AgentSandbox()

    @app.post("/v1/validate/input", response_model=ValidateResponse)
    async def validate_input(req: ValidateRequest) -> ValidateResponse:
        """Validate LLM input through guardrails."""
        result = await proxy.validate_input(req.text, req.client_id)
        return ValidateResponse(
            allowed=result.allowed,
            action=result.action,
            risk_level=result.risk_level,
            injection_score=result.injection_score,
            similarity_score=result.similarity_score,
            violations=[
                {"rule": v.rule, "severity": v.severity, "message": v.message}
                for v in result.violations
            ],
            latency_ms=result.latency_ms,
        )

    @app.post("/v1/validate/output", response_model=dict)
    async def validate_output(req: ValidateRequest) -> dict:
        """Validate LLM output before returning to user."""
        result = await proxy.validate_output(req.text)
        return {
            "allowed": result.allowed,
            "action": result.action,
            "redacted_text": result.redacted_text,
            "pii_found": result.pii_found,
            "secrets_found": result.secrets_found,
            "violations": [
                {"rule": v.rule, "severity": v.severity, "message": v.message}
                for v in result.violations
            ],
        }

    @app.post("/v1/classify", response_model=ClassifyResponse)
    async def classify(req: ClassifyRequest) -> ClassifyResponse:
        """Classify text for prompt injection."""
        result = classifier.classify_detailed(req.text)
        return ClassifyResponse(
            score=result.score,
            label=result.label,
            signals=result.signals,
            matched_patterns=result.matched_patterns,
        )

    @app.post("/v1/validate/tool", response_model=ToolCallResponse)
    async def validate_tool(req: ToolCallRequest) -> ToolCallResponse:
        """Validate an agent tool call."""
        allowed, violations = sandbox.validate_tool_call(req.tool_name, req.args)
        return ToolCallResponse(
            allowed=allowed,
            violations=[
                {"type": v.violation_type, "severity": v.severity, "message": v.message}
                for v in violations
            ],
        )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "healthy", "service": "aegis", "version": "0.1.0"}

    return app
