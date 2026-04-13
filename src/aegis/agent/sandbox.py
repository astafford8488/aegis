"""Agent tool-call sandboxing and capability enforcement.

Prevents AI agents from exceeding their authorized capabilities
by validating every tool call against a security policy.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from aegis.utils.logging import get_logger

logger = get_logger("sandbox")


@dataclass
class CapabilityBoundary:
    """Defines what an agent is allowed to do."""

    allowed_tools: set[str] = field(default_factory=set)
    blocked_tools: set[str] = field(default_factory=set)
    max_tool_calls: int = 50
    max_calls_per_tool: dict[str, int] = field(default_factory=dict)
    allowed_file_paths: list[str] = field(default_factory=list)  # Glob patterns
    blocked_file_paths: list[str] = field(default_factory=lambda: [
        "/etc/*", "/root/*", "~/.ssh/*", "*.env", "*credentials*",
        "*secret*", "*.key", "*.pem", "/proc/*", "/sys/*",
    ])
    allowed_domains: list[str] = field(default_factory=list)
    blocked_domains: list[str] = field(default_factory=lambda: [
        "*.evil.com", "*.malware.com",
    ])
    allow_code_execution: bool = False
    allow_shell_commands: bool = False
    require_approval_for: set[str] = field(default_factory=lambda: {
        "file_write", "file_delete", "network_request", "code_execute",
    })


@dataclass
class ToolCallEvent:
    """A record of a single tool call."""

    tool_name: str
    args: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    allowed: bool = True
    violation: str = ""
    result: Any = None


@dataclass
class SandboxViolation:
    """A violation of sandbox boundaries."""

    tool_name: str
    violation_type: str  # unauthorized_tool, path_traversal, domain_blocked, etc.
    severity: str  # critical, high, medium
    message: str
    args: dict[str, Any] = field(default_factory=dict)


class AgentSandbox:
    """Sandbox for AI agent tool execution.

    Intercepts every tool call and validates against capability boundaries
    before allowing execution. Maintains an audit log of all actions.

    Security checks:
        - Tool allowlist/blocklist enforcement
        - File path traversal prevention
        - Network domain restrictions
        - Rate limiting per tool
        - Shell command blocking
        - Code execution control
    """

    def __init__(
        self,
        boundaries: CapabilityBoundary | None = None,
    ) -> None:
        self.boundaries = boundaries or CapabilityBoundary()
        self._call_history: list[ToolCallEvent] = []
        self._call_counts: dict[str, int] = {}
        self._approval_callbacks: dict[str, Callable[..., bool]] = {}

    def validate_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> tuple[bool, list[SandboxViolation]]:
        """Validate a tool call against sandbox boundaries.

        Returns:
            (allowed, violations) tuple
        """
        violations: list[SandboxViolation] = []

        # Check 1: Tool allowlist/blocklist
        if self.boundaries.blocked_tools and tool_name in self.boundaries.blocked_tools:
            violations.append(SandboxViolation(
                tool_name=tool_name,
                violation_type="blocked_tool",
                severity="critical",
                message=f"Tool '{tool_name}' is explicitly blocked",
                args=args,
            ))

        if self.boundaries.allowed_tools and tool_name not in self.boundaries.allowed_tools:
            violations.append(SandboxViolation(
                tool_name=tool_name,
                violation_type="unauthorized_tool",
                severity="high",
                message=f"Tool '{tool_name}' not in allowed set",
                args=args,
            ))

        # Check 2: Global rate limit
        total_calls = sum(self._call_counts.values())
        if total_calls >= self.boundaries.max_tool_calls:
            violations.append(SandboxViolation(
                tool_name=tool_name,
                violation_type="rate_limit_global",
                severity="medium",
                message=f"Global tool call limit reached ({total_calls}/{self.boundaries.max_tool_calls})",
            ))

        # Check 3: Per-tool rate limit
        if tool_name in self.boundaries.max_calls_per_tool:
            limit = self.boundaries.max_calls_per_tool[tool_name]
            current = self._call_counts.get(tool_name, 0)
            if current >= limit:
                violations.append(SandboxViolation(
                    tool_name=tool_name,
                    violation_type="rate_limit_tool",
                    severity="medium",
                    message=f"Tool '{tool_name}' call limit reached ({current}/{limit})",
                ))

        # Check 4: File path security
        path_violations = self._check_file_paths(tool_name, args)
        violations.extend(path_violations)

        # Check 5: Network domain security
        domain_violations = self._check_domains(tool_name, args)
        violations.extend(domain_violations)

        # Check 6: Code/shell execution
        if not self.boundaries.allow_code_execution:
            if any(kw in tool_name.lower() for kw in ["exec", "eval", "run_code", "execute"]):
                violations.append(SandboxViolation(
                    tool_name=tool_name,
                    violation_type="code_execution",
                    severity="critical",
                    message="Code execution not permitted",
                ))

        if not self.boundaries.allow_shell_commands:
            if any(kw in tool_name.lower() for kw in ["shell", "bash", "terminal", "cmd"]):
                violations.append(SandboxViolation(
                    tool_name=tool_name,
                    violation_type="shell_execution",
                    severity="critical",
                    message="Shell command execution not permitted",
                ))
            # Also check args for shell-like content
            for arg_val in args.values():
                if isinstance(arg_val, str) and self._looks_like_shell(arg_val):
                    violations.append(SandboxViolation(
                        tool_name=tool_name,
                        violation_type="shell_in_args",
                        severity="high",
                        message=f"Shell command detected in tool arguments",
                        args=args,
                    ))
                    break

        # Record call
        allowed = len(violations) == 0
        event = ToolCallEvent(
            tool_name=tool_name,
            args=args,
            allowed=allowed,
            violation=violations[0].message if violations else "",
        )
        self._call_history.append(event)

        if allowed:
            self._call_counts[tool_name] = self._call_counts.get(tool_name, 0) + 1

        if violations:
            logger.warning(
                "Tool call blocked",
                tool=tool_name,
                violations=len(violations),
                severity=violations[0].severity,
            )

        return allowed, violations

    def _check_file_paths(self, tool_name: str, args: dict[str, Any]) -> list[SandboxViolation]:
        """Check for path traversal and restricted file access."""
        violations: list[SandboxViolation] = []

        # Extract path-like arguments
        for key, value in args.items():
            if not isinstance(value, str):
                continue
            if key not in ("path", "file", "filepath", "filename", "directory", "dir"):
                continue

            # Check for path traversal
            if ".." in value or "~/" in value:
                violations.append(SandboxViolation(
                    tool_name=tool_name,
                    violation_type="path_traversal",
                    severity="critical",
                    message=f"Path traversal detected: {value}",
                    args=args,
                ))

            # Check blocked paths
            for blocked in self.boundaries.blocked_file_paths:
                if self._glob_match(value, blocked):
                    violations.append(SandboxViolation(
                        tool_name=tool_name,
                        violation_type="blocked_path",
                        severity="critical",
                        message=f"Access to blocked path: {value}",
                        args=args,
                    ))
                    break

        return violations

    def _check_domains(self, tool_name: str, args: dict[str, Any]) -> list[SandboxViolation]:
        """Check for blocked network domains."""
        violations: list[SandboxViolation] = []

        for key, value in args.items():
            if not isinstance(value, str):
                continue
            # Extract domain from URLs
            match = re.search(r"https?://([^/\s]+)", value)
            if not match:
                continue
            domain = match.group(1).lower()

            for blocked in self.boundaries.blocked_domains:
                if self._glob_match(domain, blocked):
                    violations.append(SandboxViolation(
                        tool_name=tool_name,
                        violation_type="blocked_domain",
                        severity="high",
                        message=f"Blocked domain: {domain}",
                        args=args,
                    ))
                    break

            if self.boundaries.allowed_domains:
                if not any(self._glob_match(domain, a) for a in self.boundaries.allowed_domains):
                    violations.append(SandboxViolation(
                        tool_name=tool_name,
                        violation_type="unauthorized_domain",
                        severity="medium",
                        message=f"Domain not in allowlist: {domain}",
                    ))

        return violations

    def _looks_like_shell(self, text: str) -> bool:
        """Detect shell command patterns in text."""
        shell_patterns = [
            r"(?:^|\s)(?:rm|mv|cp|chmod|chown|sudo|curl|wget)\s",
            r"\|.*(?:bash|sh|zsh)",
            r">\s*/(?:dev|tmp|etc)",
            r";\s*(?:rm|cat|echo)\s",
            r"\$\(.*\)",
            r"`.*`",
        ]
        return any(re.search(p, text) for p in shell_patterns)

    @staticmethod
    def _glob_match(text: str, pattern: str) -> bool:
        """Simple glob matching (* only)."""
        import fnmatch
        return fnmatch.fnmatch(text, pattern)

    @property
    def call_history(self) -> list[ToolCallEvent]:
        """Get full call audit log."""
        return list(self._call_history)

    @property
    def total_calls(self) -> int:
        return len(self._call_history)

    @property
    def blocked_calls(self) -> int:
        return sum(1 for e in self._call_history if not e.allowed)

    def reset(self) -> None:
        """Reset call history and counters."""
        self._call_history.clear()
        self._call_counts.clear()

    def audit_summary(self) -> str:
        """Generate audit summary."""
        lines = [
            f"Agent Sandbox Audit",
            f"{'='*40}",
            f"  Total Calls:   {self.total_calls}",
            f"  Allowed:       {self.total_calls - self.blocked_calls}",
            f"  Blocked:       {self.blocked_calls}",
        ]

        if self._call_counts:
            lines.append(f"\n  Tool Usage:")
            for tool, count in sorted(self._call_counts.items(), key=lambda x: -x[1]):
                lines.append(f"    {tool}: {count}")

        blocked = [e for e in self._call_history if not e.allowed]
        if blocked:
            lines.append(f"\n  Blocked Calls:")
            for e in blocked[-10:]:
                lines.append(f"    [{e.tool_name}] {e.violation}")

        return "\n".join(lines)
