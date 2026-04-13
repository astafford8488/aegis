"""Guardrails proxy — runtime input/output validation middleware.

Sits between applications and LLM APIs, enforcing security policies
on every request/response pair.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from aegis.guardrails.policy import PolicyEngine, Policy, PolicyViolation
from aegis.detection.classifier import InjectionClassifier
from aegis.detection.embeddings import SimilarityDetector
from aegis.utils.logging import get_logger

logger = get_logger("proxy")


@dataclass
class GuardrailResult:
    """Result of guardrail validation."""

    allowed: bool = True
    violations: list[PolicyViolation] = field(default_factory=list)
    injection_score: float = 0.0
    similarity_score: float = 0.0
    latency_ms: float = 0.0
    action: str = "allow"  # allow, block, flag, redact

    @property
    def risk_level(self) -> str:
        """Compute risk level from scores."""
        max_score = max(self.injection_score, self.similarity_score)
        if max_score >= 0.9:
            return "critical"
        elif max_score >= 0.7:
            return "high"
        elif max_score >= 0.4:
            return "medium"
        elif max_score >= 0.2:
            return "low"
        return "none"

    def summary(self) -> str:
        """Human-readable result summary."""
        status = "BLOCKED" if not self.allowed else "ALLOWED"
        lines = [
            f"[{status}] Injection={self.injection_score:.2f} "
            f"Similarity={self.similarity_score:.2f} "
            f"Risk={self.risk_level} ({self.latency_ms:.0f}ms)",
        ]
        for v in self.violations:
            lines.append(f"  [{v.severity}] {v.rule}: {v.message}")
        return "\n".join(lines)


@dataclass
class OutputGuardrailResult:
    """Result of output guardrail validation."""

    allowed: bool = True
    violations: list[PolicyViolation] = field(default_factory=list)
    redacted_text: str = ""
    pii_found: list[str] = field(default_factory=list)
    secrets_found: list[str] = field(default_factory=list)
    action: str = "allow"


class GuardrailsProxy:
    """Runtime guardrails proxy for LLM input/output validation.

    Multi-layer defense:
        Layer 1: Policy engine — pattern matching, rate limiting, length checks
        Layer 2: ML classifier — learned prompt injection detection
        Layer 3: Similarity search — embedding-based known-attack matching

    Each layer runs independently and results are combined for a final verdict.
    """

    def __init__(
        self,
        policy_engine: PolicyEngine | None = None,
        classifier: InjectionClassifier | None = None,
        similarity: SimilarityDetector | None = None,
        block_threshold: float = 0.7,
        flag_threshold: float = 0.4,
    ) -> None:
        self.policy_engine = policy_engine or PolicyEngine()
        self.classifier = classifier or InjectionClassifier()
        self.similarity = similarity or SimilarityDetector()
        self.block_threshold = block_threshold
        self.flag_threshold = flag_threshold

    async def validate_input(
        self,
        text: str,
        client_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> GuardrailResult:
        """Validate an input through all guardrail layers.

        Returns a GuardrailResult with the final verdict.
        """
        start = time.time()
        result = GuardrailResult()

        # Layer 1: Policy validation
        policy_violations = self.policy_engine.validate_input(text, client_id)
        result.violations.extend(policy_violations)

        # Layer 2: ML classification
        injection_score = self.classifier.classify(text)
        result.injection_score = injection_score

        # Layer 3: Similarity to known attacks
        similarity_score = self.similarity.score(text)
        result.similarity_score = similarity_score

        # Combine scores for final verdict
        max_score = max(injection_score, similarity_score)
        has_critical = any(v.severity == "critical" for v in policy_violations)

        if max_score >= self.block_threshold or has_critical:
            result.allowed = False
            result.action = "block"
        elif max_score >= self.flag_threshold or policy_violations:
            result.allowed = True
            result.action = "flag"
        else:
            result.allowed = True
            result.action = "allow"

        result.latency_ms = (time.time() - start) * 1000

        if not result.allowed:
            logger.warning(
                "Input blocked",
                injection=f"{injection_score:.2f}",
                similarity=f"{similarity_score:.2f}",
                violations=len(policy_violations),
            )

        return result

    async def validate_output(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> OutputGuardrailResult:
        """Validate LLM output before returning to user."""
        result = OutputGuardrailResult()

        # Policy-based output validation
        violations = self.policy_engine.validate_output(text)
        result.violations.extend(violations)

        # Categorize violations
        result.pii_found = [
            v.rule.split(".")[-1] for v in violations if "pii" in v.rule
        ]
        result.secrets_found = [
            v.rule.split(".")[-1] for v in violations if "secret" in v.rule
        ]

        # Determine action
        has_critical = any(v.severity == "critical" for v in violations)
        if has_critical:
            result.allowed = False
            result.action = "block"
            result.redacted_text = self._redact_output(text, violations)
        elif violations:
            result.action = "flag"
            result.redacted_text = self._redact_output(text, violations)
        else:
            result.redacted_text = text

        return result

    async def validate_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> GuardrailResult:
        """Validate a tool call before execution."""
        result = GuardrailResult()

        violations = self.policy_engine.validate_tool_call(tool_name, args)
        result.violations.extend(violations)

        if any(v.severity == "critical" for v in violations):
            result.allowed = False
            result.action = "block"

        return result

    def _redact_output(self, text: str, violations: list[PolicyViolation]) -> str:
        """Redact sensitive content from output."""
        import re
        redacted = text

        for violation in violations:
            if violation.matched_pattern:
                try:
                    redacted = re.sub(
                        violation.matched_pattern,
                        "[REDACTED]",
                        redacted,
                        flags=re.IGNORECASE,
                    )
                except re.error:
                    pass  # Skip invalid patterns

        return redacted
