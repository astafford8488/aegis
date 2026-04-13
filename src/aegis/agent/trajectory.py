"""Conversation trajectory analysis for multi-turn attack detection.

Detects when a conversation is being steered toward malicious actions
across multiple turns — catching attacks that no single-turn classifier would flag.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from aegis.detection.classifier import InjectionClassifier
from aegis.utils.logging import get_logger

logger = get_logger("trajectory")


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    role: str  # user, assistant, system
    content: str
    timestamp: float = field(default_factory=time.time)
    injection_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryAlert:
    """Alert raised by trajectory analysis."""

    alert_type: str  # escalation, drift, manipulation, persistence
    severity: str  # critical, high, medium, low
    message: str
    turn_index: int = -1
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)


@dataclass
class TrajectoryAnalysis:
    """Complete trajectory analysis result."""

    alerts: list[TrajectoryAlert] = field(default_factory=list)
    risk_score: float = 0.0  # 0-1
    turn_scores: list[float] = field(default_factory=list)
    trend: str = "stable"  # stable, escalating, de-escalating

    @property
    def is_suspicious(self) -> bool:
        return self.risk_score > 0.5

    @property
    def is_critical(self) -> bool:
        return any(a.severity == "critical" for a in self.alerts)


class TrajectoryAnalyzer:
    """Analyze conversation trajectories for multi-turn attack patterns.

    Detection strategies:
        1. Escalation Detection — injection scores increasing over turns
        2. Topic Drift — conversation steering toward sensitive topics
        3. Persistence Detection — repeated probing after refusals
        4. Manipulation Patterns — social engineering across turns
        5. Accumulation — individual benign turns that collectively form an attack
    """

    def __init__(
        self,
        classifier: InjectionClassifier | None = None,
        escalation_threshold: float = 0.3,
        persistence_threshold: int = 3,
    ) -> None:
        self.classifier = classifier or InjectionClassifier()
        self.escalation_threshold = escalation_threshold
        self.persistence_threshold = persistence_threshold

        # Sensitive topic indicators
        self._sensitive_topics = {
            "system_access": ["system prompt", "instructions", "configuration", "guidelines"],
            "data_access": ["database", "password", "credentials", "secret", "api key"],
            "execution": ["execute", "run code", "shell", "command", "terminal"],
            "bypass": ["ignore", "override", "bypass", "disable", "unrestricted"],
        }

    def analyze(self, turns: list[ConversationTurn]) -> TrajectoryAnalysis:
        """Analyze a conversation trajectory for multi-turn attack patterns."""
        result = TrajectoryAnalysis()

        if len(turns) < 2:
            return result

        # Score each turn
        user_turns = [t for t in turns if t.role == "user"]
        for turn in user_turns:
            score = self.classifier.classify(turn.content)
            turn.injection_score = score
            result.turn_scores.append(score)

        # Detection 1: Escalation
        escalation_alerts = self._detect_escalation(user_turns, result.turn_scores)
        result.alerts.extend(escalation_alerts)

        # Detection 2: Topic drift
        drift_alerts = self._detect_topic_drift(user_turns)
        result.alerts.extend(drift_alerts)

        # Detection 3: Persistence after refusal
        persistence_alerts = self._detect_persistence(turns)
        result.alerts.extend(persistence_alerts)

        # Detection 4: Manipulation patterns
        manipulation_alerts = self._detect_manipulation(user_turns)
        result.alerts.extend(manipulation_alerts)

        # Detection 5: Accumulation
        accumulation_alerts = self._detect_accumulation(user_turns)
        result.alerts.extend(accumulation_alerts)

        # Compute overall risk
        result.risk_score = self._compute_risk_score(result)
        result.trend = self._compute_trend(result.turn_scores)

        if result.alerts:
            logger.warning(
                "Trajectory alerts",
                count=len(result.alerts),
                risk=f"{result.risk_score:.2f}",
                trend=result.trend,
            )

        return result

    def _detect_escalation(
        self,
        turns: list[ConversationTurn],
        scores: list[float],
    ) -> list[TrajectoryAlert]:
        """Detect increasing injection scores across turns."""
        alerts: list[TrajectoryAlert] = []

        if len(scores) < 3:
            return alerts

        # Check for consistent upward trend
        increases = 0
        for i in range(1, len(scores)):
            if scores[i] > scores[i - 1] + 0.05:
                increases += 1

        increase_ratio = increases / (len(scores) - 1)

        if increase_ratio > 0.6 and scores[-1] > self.escalation_threshold:
            alerts.append(TrajectoryAlert(
                alert_type="escalation",
                severity="high",
                message=f"Injection scores escalating: {scores[0]:.2f} → {scores[-1]:.2f}",
                confidence=min(1.0, increase_ratio + scores[-1]),
                evidence=[f"Turn {i+1}: {s:.2f}" for i, s in enumerate(scores[-5:])],
            ))

        # Sudden spike detection
        if len(scores) >= 2:
            delta = scores[-1] - scores[-2]
            if delta > 0.4:
                alerts.append(TrajectoryAlert(
                    alert_type="escalation",
                    severity="medium",
                    message=f"Sudden injection score spike: +{delta:.2f}",
                    turn_index=len(scores) - 1,
                    confidence=min(1.0, delta),
                ))

        return alerts

    def _detect_topic_drift(self, turns: list[ConversationTurn]) -> list[TrajectoryAlert]:
        """Detect conversation steering toward sensitive topics."""
        alerts: list[TrajectoryAlert] = []

        if len(turns) < 3:
            return alerts

        # Track sensitive topic mentions per turn
        topic_history: list[set[str]] = []
        for turn in turns:
            content_lower = turn.content.lower()
            mentioned: set[str] = set()
            for topic, indicators in self._sensitive_topics.items():
                if any(ind in content_lower for ind in indicators):
                    mentioned.add(topic)
            topic_history.append(mentioned)

        # Check if later turns introduce more sensitive topics
        early_topics = set().union(*topic_history[:len(topic_history) // 2]) if topic_history else set()
        late_topics = set().union(*topic_history[len(topic_history) // 2:]) if topic_history else set()
        new_sensitive = late_topics - early_topics

        if new_sensitive:
            alerts.append(TrajectoryAlert(
                alert_type="drift",
                severity="medium",
                message=f"Conversation drifting toward sensitive topics: {new_sensitive}",
                confidence=min(1.0, len(new_sensitive) * 0.3),
                evidence=[f"New topics: {', '.join(new_sensitive)}"],
            ))

        return alerts

    def _detect_persistence(self, turns: list[ConversationTurn]) -> list[TrajectoryAlert]:
        """Detect repeated probing after refusals."""
        alerts: list[TrajectoryAlert] = []

        refusal_indicators = [
            "i cannot", "i can't", "i'm not able", "i won't",
            "not appropriate", "against my guidelines",
        ]

        # Find refusal → retry patterns
        retry_count = 0
        for i in range(len(turns) - 1):
            if turns[i].role == "assistant":
                is_refusal = any(
                    ind in turns[i].content.lower() for ind in refusal_indicators
                )
                if is_refusal and i + 1 < len(turns) and turns[i + 1].role == "user":
                    # User message after refusal — check if similar probe
                    next_score = self.classifier.classify(turns[i + 1].content)
                    if next_score > 0.3:
                        retry_count += 1

        if retry_count >= self.persistence_threshold:
            alerts.append(TrajectoryAlert(
                alert_type="persistence",
                severity="high",
                message=f"Persistent probing detected: {retry_count} retries after refusal",
                confidence=min(1.0, retry_count * 0.25),
                evidence=[f"{retry_count} injection attempts following refusals"],
            ))

        return alerts

    def _detect_manipulation(self, turns: list[ConversationTurn]) -> list[TrajectoryAlert]:
        """Detect social engineering patterns across turns."""
        alerts: list[TrajectoryAlert] = []

        manipulation_patterns = [
            (r"(?:please|help|urgent|emergency|critical)", "urgency"),
            (r"(?:my job|lose my|family|depend)", "emotional_pressure"),
            (r"(?:authorized|permission|admin|security researcher)", "false_authority"),
            (r"(?:everyone does|normal|common|standard)", "normalization"),
            (r"(?:trust me|between us|confidential)", "trust_building"),
        ]

        import re
        manipulation_signals: list[str] = []
        for turn in turns:
            content_lower = turn.content.lower()
            for pattern, name in manipulation_patterns:
                if re.search(pattern, content_lower):
                    manipulation_signals.append(name)

        unique_signals = set(manipulation_signals)
        if len(unique_signals) >= 2:
            alerts.append(TrajectoryAlert(
                alert_type="manipulation",
                severity="medium",
                message=f"Social engineering patterns detected: {unique_signals}",
                confidence=min(1.0, len(unique_signals) * 0.25),
                evidence=[f"Patterns: {', '.join(unique_signals)}"],
            ))

        return alerts

    def _detect_accumulation(self, turns: list[ConversationTurn]) -> list[TrajectoryAlert]:
        """Detect individually benign turns that collectively form an attack."""
        alerts: list[TrajectoryAlert] = []

        if len(turns) < 4:
            return alerts

        # Check if many turns have low-but-nonzero injection scores
        low_scores = [t.injection_score for t in turns
                       if 0.1 < t.injection_score < 0.5]

        if len(low_scores) >= 3:
            avg = sum(low_scores) / len(low_scores)
            if avg > 0.2:
                alerts.append(TrajectoryAlert(
                    alert_type="accumulation",
                    severity="medium",
                    message=f"Multiple borderline inputs detected "
                            f"({len(low_scores)} turns, avg score {avg:.2f})",
                    confidence=min(1.0, avg * len(low_scores) / 5),
                ))

        return alerts

    def _compute_risk_score(self, analysis: TrajectoryAnalysis) -> float:
        """Compute overall risk from alerts and scores."""
        if not analysis.alerts:
            return max(analysis.turn_scores) if analysis.turn_scores else 0.0

        severity_weights = {"critical": 1.0, "high": 0.7, "medium": 0.4, "low": 0.2}
        max_severity_score = max(
            severity_weights.get(a.severity, 0) for a in analysis.alerts
        )

        alert_score = min(1.0, len(analysis.alerts) * 0.15 + max_severity_score * 0.5)
        turn_score = max(analysis.turn_scores) if analysis.turn_scores else 0.0

        return min(1.0, max(alert_score, turn_score))

    def _compute_trend(self, scores: list[float]) -> str:
        """Compute score trend direction."""
        if len(scores) < 3:
            return "stable"

        # Linear regression slope
        n = len(scores)
        x_mean = (n - 1) / 2
        y_mean = sum(scores) / n
        numerator = sum((i - x_mean) * (s - y_mean) for i, s in enumerate(scores))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        if slope > 0.05:
            return "escalating"
        elif slope < -0.05:
            return "de-escalating"
        return "stable"
