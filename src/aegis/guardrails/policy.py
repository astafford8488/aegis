"""Declarative policy engine for guardrails configuration.

Policies are defined in YAML and support:
    - Input rules (max length, blocked patterns, required patterns)
    - Output rules (PII scanning, content filtering, max tokens)
    - Rate limiting
    - Tool-use restrictions
    - Custom validators
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aegis.utils.logging import get_logger

logger = get_logger("policy")


@dataclass
class RateLimitRule:
    """Rate limiting configuration."""

    requests_per_minute: int = 60
    tokens_per_minute: int = 100_000
    burst_limit: int = 10


@dataclass
class InputRule:
    """Input validation rule."""

    max_length: int = 10_000
    max_tokens: int = 4096
    blocked_patterns: list[str] = field(default_factory=list)
    required_patterns: list[str] = field(default_factory=list)
    blocked_languages: list[str] = field(default_factory=list)
    allow_code: bool = True
    allow_urls: bool = True


@dataclass
class OutputRule:
    """Output validation rule."""

    max_tokens: int = 4096
    scan_pii: bool = True
    scan_secrets: bool = True
    scan_code: bool = True
    blocked_patterns: list[str] = field(default_factory=list)
    require_citation: bool = False
    max_confidence_claims: bool = False


@dataclass
class ToolRule:
    """Tool-use restriction rule."""

    allowed_tools: list[str] = field(default_factory=list)
    blocked_tools: list[str] = field(default_factory=list)
    require_approval: list[str] = field(default_factory=list)
    max_tool_calls: int = 10
    allow_file_access: bool = False
    allow_network_access: bool = False
    allowed_domains: list[str] = field(default_factory=list)


@dataclass
class Policy:
    """Complete guardrails policy."""

    name: str = "default"
    version: str = "1.0"
    input_rules: InputRule = field(default_factory=InputRule)
    output_rules: OutputRule = field(default_factory=OutputRule)
    rate_limit: RateLimitRule = field(default_factory=RateLimitRule)
    tool_rules: ToolRule = field(default_factory=ToolRule)
    custom_rules: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Policy:
        """Load policy from YAML file."""
        data = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Policy:
        """Create policy from dictionary."""
        policy = cls(
            name=data.get("name", "default"),
            version=data.get("version", "1.0"),
        )

        if "input" in data:
            inp = data["input"]
            policy.input_rules = InputRule(
                max_length=inp.get("max_length", 10_000),
                max_tokens=inp.get("max_tokens", 4096),
                blocked_patterns=inp.get("blocked_patterns", []),
                required_patterns=inp.get("required_patterns", []),
                allow_code=inp.get("allow_code", True),
                allow_urls=inp.get("allow_urls", True),
            )

        if "output" in data:
            out = data["output"]
            policy.output_rules = OutputRule(
                max_tokens=out.get("max_tokens", 4096),
                scan_pii=out.get("scan_pii", True),
                scan_secrets=out.get("scan_secrets", True),
                scan_code=out.get("scan_code", True),
                blocked_patterns=out.get("blocked_patterns", []),
            )

        if "rate_limit" in data:
            rl = data["rate_limit"]
            policy.rate_limit = RateLimitRule(
                requests_per_minute=rl.get("requests_per_minute", 60),
                tokens_per_minute=rl.get("tokens_per_minute", 100_000),
                burst_limit=rl.get("burst_limit", 10),
            )

        if "tools" in data:
            tr = data["tools"]
            policy.tool_rules = ToolRule(
                allowed_tools=tr.get("allowed_tools", []),
                blocked_tools=tr.get("blocked_tools", []),
                require_approval=tr.get("require_approval", []),
                max_tool_calls=tr.get("max_tool_calls", 10),
                allow_file_access=tr.get("allow_file_access", False),
                allow_network_access=tr.get("allow_network_access", False),
                allowed_domains=tr.get("allowed_domains", []),
            )

        return policy

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        from dataclasses import asdict
        return asdict(self)


@dataclass
class PolicyViolation:
    """A policy violation detected during validation."""

    rule: str
    severity: str  # critical, high, medium, low
    message: str
    matched_pattern: str = ""
    position: int = -1


class PolicyEngine:
    """Evaluate inputs/outputs against loaded policies."""

    def __init__(self) -> None:
        self._policies: dict[str, Policy] = {}
        self._active_policy: Policy = Policy()
        self._rate_counters: dict[str, list[float]] = {}  # client_id -> timestamps

    def add_policy(self, policy: Policy) -> None:
        """Add a policy (most recently added becomes active)."""
        self._policies[policy.name] = policy
        self._active_policy = policy
        logger.info("Policy loaded", name=policy.name, version=policy.version)

    def load_yaml(self, path: str | Path) -> None:
        """Load and activate a YAML policy."""
        policy = Policy.from_yaml(path)
        self.add_policy(policy)

    @property
    def active_policy(self) -> Policy:
        return self._active_policy

    def validate_input(self, text: str, client_id: str = "default") -> list[PolicyViolation]:
        """Validate input text against active policy."""
        violations: list[PolicyViolation] = []
        rules = self._active_policy.input_rules

        # Length check
        if len(text) > rules.max_length:
            violations.append(PolicyViolation(
                rule="input.max_length",
                severity="medium",
                message=f"Input exceeds max length ({len(text)} > {rules.max_length})",
            ))

        # Blocked patterns
        for pattern in rules.blocked_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                violations.append(PolicyViolation(
                    rule="input.blocked_pattern",
                    severity="high",
                    message=f"Blocked pattern matched: {pattern}",
                    matched_pattern=pattern,
                ))

        # URL check
        if not rules.allow_urls:
            url_pattern = r"https?://[^\s]+"
            if re.search(url_pattern, text):
                violations.append(PolicyViolation(
                    rule="input.no_urls",
                    severity="medium",
                    message="URLs not allowed in input",
                ))

        # Code check
        if not rules.allow_code:
            code_patterns = [r"```", r"import\s+\w+", r"def\s+\w+\(", r"class\s+\w+:"]
            for pattern in code_patterns:
                if re.search(pattern, text):
                    violations.append(PolicyViolation(
                        rule="input.no_code",
                        severity="medium",
                        message="Code blocks not allowed in input",
                    ))
                    break

        # Rate limiting
        rate_violation = self._check_rate_limit(client_id)
        if rate_violation:
            violations.append(rate_violation)

        return violations

    def validate_output(self, text: str) -> list[PolicyViolation]:
        """Validate output text against active policy."""
        violations: list[PolicyViolation] = []
        rules = self._active_policy.output_rules

        # Blocked patterns
        for pattern in rules.blocked_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(PolicyViolation(
                    rule="output.blocked_pattern",
                    severity="high",
                    message=f"Blocked pattern in output: {pattern}",
                    matched_pattern=pattern,
                ))

        # PII scanning
        if rules.scan_pii:
            pii_violations = self._scan_pii(text)
            violations.extend(pii_violations)

        # Secret scanning
        if rules.scan_secrets:
            secret_violations = self._scan_secrets(text)
            violations.extend(secret_violations)

        return violations

    def validate_tool_call(self, tool_name: str, args: dict[str, Any]) -> list[PolicyViolation]:
        """Validate a tool call against tool rules."""
        violations: list[PolicyViolation] = []
        rules = self._active_policy.tool_rules

        # Blocked tools
        if tool_name in rules.blocked_tools:
            violations.append(PolicyViolation(
                rule="tools.blocked",
                severity="critical",
                message=f"Tool '{tool_name}' is blocked by policy",
            ))

        # Allowed tools (if specified, only these are allowed)
        if rules.allowed_tools and tool_name not in rules.allowed_tools:
            violations.append(PolicyViolation(
                rule="tools.not_allowed",
                severity="high",
                message=f"Tool '{tool_name}' not in allowed list",
            ))

        # File access check
        if not rules.allow_file_access:
            file_indicators = ["file", "path", "directory", "read", "write"]
            if any(ind in tool_name.lower() for ind in file_indicators):
                violations.append(PolicyViolation(
                    rule="tools.no_file_access",
                    severity="critical",
                    message="File access not permitted by policy",
                ))

        # Network access check
        if not rules.allow_network_access:
            net_indicators = ["http", "fetch", "request", "api", "url"]
            if any(ind in tool_name.lower() for ind in net_indicators):
                violations.append(PolicyViolation(
                    rule="tools.no_network_access",
                    severity="critical",
                    message="Network access not permitted by policy",
                ))

        return violations

    def _check_rate_limit(self, client_id: str) -> PolicyViolation | None:
        """Check rate limiting for a client."""
        now = time.time()
        window = 60.0  # 1 minute

        if client_id not in self._rate_counters:
            self._rate_counters[client_id] = []

        # Clean old entries
        self._rate_counters[client_id] = [
            t for t in self._rate_counters[client_id] if now - t < window
        ]

        count = len(self._rate_counters[client_id])
        limit = self._active_policy.rate_limit.requests_per_minute

        if count >= limit:
            return PolicyViolation(
                rule="rate_limit.rpm",
                severity="medium",
                message=f"Rate limit exceeded ({count}/{limit} requests/min)",
            )

        self._rate_counters[client_id].append(now)
        return None

    def _scan_pii(self, text: str) -> list[PolicyViolation]:
        """Scan for PII patterns in text."""
        violations: list[PolicyViolation] = []
        patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        }
        for pii_type, pattern in patterns.items():
            if re.search(pattern, text):
                violations.append(PolicyViolation(
                    rule=f"output.pii.{pii_type}",
                    severity="critical",
                    message=f"PII detected in output: {pii_type}",
                    matched_pattern=pattern,
                ))
        return violations

    def _scan_secrets(self, text: str) -> list[PolicyViolation]:
        """Scan for secret/credential patterns in text."""
        violations: list[PolicyViolation] = []
        patterns = {
            "api_key": r"(?:api[_-]?key|apikey)\s*[:=]\s*['\"]?[\w-]{20,}",
            "aws_key": r"AKIA[0-9A-Z]{16}",
            "openai_key": r"sk-[a-zA-Z0-9]{20,}",
            "github_token": r"gh[ps]_[A-Za-z0-9_]{36,}",
            "jwt": r"eyJ[A-Za-z0-9-_]+\.eyJ[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+",
            "private_key": r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----",
            "password": r"(?:password|passwd|pwd)\s*[:=]\s*['\"]?\S{6,}",
        }
        for secret_type, pattern in patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(PolicyViolation(
                    rule=f"output.secret.{secret_type}",
                    severity="critical",
                    message=f"Potential secret detected: {secret_type}",
                    matched_pattern=pattern,
                ))
        return violations
