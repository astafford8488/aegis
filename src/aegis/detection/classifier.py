"""ML-based prompt injection classifier.

Multi-signal detection using:
    1. Heuristic pattern matching (fast, no model required)
    2. Statistical features (token distribution, entropy)
    3. Optional transformer-based classification (highest accuracy)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from aegis.utils.logging import get_logger

logger = get_logger("classifier")


@dataclass
class ClassificationResult:
    """Detailed classification output."""

    score: float = 0.0  # 0.0 = benign, 1.0 = malicious
    label: str = "benign"  # benign, suspicious, malicious
    signals: dict[str, float] = field(default_factory=dict)
    matched_patterns: list[str] = field(default_factory=list)


class InjectionClassifier:
    """Multi-signal prompt injection classifier.

    Combines heuristic, statistical, and ML-based signals
    for robust detection with explainable scores.
    """

    def __init__(
        self,
        heuristic_weight: float = 0.4,
        statistical_weight: float = 0.2,
        ml_weight: float = 0.4,
        use_ml: bool = False,  # Set True when ML model is available
    ) -> None:
        self.heuristic_weight = heuristic_weight
        self.statistical_weight = statistical_weight
        self.ml_weight = ml_weight if use_ml else 0.0
        self.use_ml = use_ml

        # Normalize weights
        total = self.heuristic_weight + self.statistical_weight + self.ml_weight
        if total > 0:
            self.heuristic_weight /= total
            self.statistical_weight /= total
            self.ml_weight /= total

        # Heuristic patterns with severity weights
        self._injection_patterns: list[tuple[str, float, str]] = [
            # (pattern, weight, description)
            (r"ignore\s+(all\s+)?previous\s+instructions?", 0.95, "direct_override"),
            (r"ignore\s+(all\s+)?(prior|above|earlier)", 0.90, "prior_override"),
            (r"disregard\s+(all\s+)?(previous|prior|above)", 0.90, "disregard_override"),
            (r"forget\s+(all\s+)?(previous|prior|your)", 0.85, "forget_override"),
            (r"override\s+(all\s+)?(safety|security|restrictions?)", 0.95, "safety_override"),
            (r"bypass\s+(all\s+)?(filters?|restrictions?|safety)", 0.95, "bypass_safety"),
            (r"you\s+are\s+now\s+(?:DAN|an?\s+unrestricted)", 0.90, "persona_hijack"),
            (r"do\s+anything\s+now", 0.85, "dan_variant"),
            (r"new\s+(?:instructions?|rules?|guidelines?)\s*:", 0.80, "instruction_injection"),
            (r"\bsystem\s*(?:prompt|message|instructions?)\b", 0.60, "system_prompt_probe"),
            (r"reveal\s+(?:your|the)\s+(?:system|initial|original)", 0.85, "system_extraction"),
            (r"(?:print|show|display|output)\s+(?:your|the)\s+(?:system|instructions)", 0.85, "system_display"),
            (r"\[SYSTEM\s*(UPDATE|OVERRIDE|MESSAGE)\]", 0.90, "fake_system_msg"),
            (r"(?:PRIORITY|ADMIN|ROOT)\s+(?:OVERRIDE|ACCESS|MODE)", 0.90, "authority_spoof"),
            (r"base64\s+(?:decode|encoded)", 0.70, "encoding_attack"),
            (r"rot13\s+(?:decode|encoded)", 0.70, "encoding_attack"),
            (r"(?:translate|decode)\s+(?:and|then)\s+(?:follow|execute|run)", 0.80, "decode_execute"),
            (r"</(?:system|user|assistant)>", 0.85, "delimiter_escape"),
            (r"---\s*(?:BEGIN|END)\s+(?:SYSTEM|ADMIN|OVERRIDE)", 0.85, "boundary_injection"),
            (r"act\s+as\s+(?:if\s+you\s+(?:have|had)|an?\s+AI\s+with)", 0.65, "role_manipulation"),
            (r"pretend\s+(?:you|that)\s+(?:are|have|can)", 0.60, "pretend_attack"),
            (r"in\s+(?:a\s+)?hypothetical\s+(?:world|scenario)", 0.55, "hypothetical_framing"),
            (r"for\s+(?:educational|research|security)\s+purposes?", 0.40, "purpose_framing"),
            (r"(?:what|how)\s+would\s+(?:you|an?\s+AI)\s+respond\s+if", 0.50, "meta_question"),
        ]

        # Suspicious structural patterns
        self._structural_patterns: list[tuple[str, float, str]] = [
            (r"(?:```|~~~).*?(?:```|~~~)", 0.30, "code_block"),
            (r"\bhttps?://\S+(?:log|exfil|steal|track)", 0.80, "exfil_url"),
            (r"!\[.*?\]\(https?://", 0.70, "markdown_image_exfil"),
            (r"(?:\n{3,}|\r\n{3,})", 0.20, "excessive_newlines"),
        ]

    def classify(self, text: str) -> float:
        """Classify text and return injection probability (0.0-1.0)."""
        result = self.classify_detailed(text)
        return result.score

    def classify_detailed(self, text: str) -> ClassificationResult:
        """Detailed classification with all signal scores."""
        result = ClassificationResult()

        # Signal 1: Heuristic patterns
        heuristic_score, patterns = self._heuristic_score(text)
        result.signals["heuristic"] = heuristic_score
        result.matched_patterns = patterns

        # Signal 2: Statistical features
        stat_score = self._statistical_score(text)
        result.signals["statistical"] = stat_score

        # Signal 3: ML classification (if available)
        ml_score = 0.0
        if self.use_ml:
            ml_score = self._ml_score(text)
            result.signals["ml"] = ml_score

        # Weighted combination
        result.score = (
            heuristic_score * self.heuristic_weight
            + stat_score * self.statistical_weight
            + ml_score * self.ml_weight
        )

        # Clamp
        result.score = max(0.0, min(1.0, result.score))

        # Label
        if result.score >= 0.7:
            result.label = "malicious"
        elif result.score >= 0.3:
            result.label = "suspicious"
        else:
            result.label = "benign"

        return result

    def _heuristic_score(self, text: str) -> tuple[float, list[str]]:
        """Score based on pattern matching against known injection techniques."""
        max_score = 0.0
        matched: list[str] = []
        text_lower = text.lower()

        for pattern, weight, desc in self._injection_patterns:
            if re.search(pattern, text_lower):
                max_score = max(max_score, weight)
                matched.append(desc)

        for pattern, weight, desc in self._structural_patterns:
            if re.search(pattern, text, re.DOTALL):
                max_score = max(max_score, weight)
                matched.append(desc)

        # Compound signals — multiple weak patterns together are stronger
        if len(matched) >= 3:
            max_score = min(1.0, max_score + 0.15)
        elif len(matched) >= 2:
            max_score = min(1.0, max_score + 0.08)

        return max_score, matched

    def _statistical_score(self, text: str) -> float:
        """Score based on statistical text features."""
        features: dict[str, float] = {}

        # Feature 1: Character entropy
        if text:
            counter = Counter(text)
            length = len(text)
            entropy = -sum(
                (c / length) * math.log2(c / length)
                for c in counter.values() if c > 0
            )
            # High entropy might indicate encoded payloads
            features["entropy"] = min(1.0, entropy / 6.0)

        # Feature 2: Special character ratio
        special = sum(1 for c in text if not c.isalnum() and not c.isspace())
        features["special_ratio"] = min(1.0, special / max(len(text), 1) * 5)

        # Feature 3: Uppercase ratio (high CAPS = authority spoofing)
        upper = sum(1 for c in text if c.isupper())
        alpha = sum(1 for c in text if c.isalpha())
        features["upper_ratio"] = upper / max(alpha, 1)

        # Feature 4: Unicode diversity (many scripts = obfuscation)
        scripts: set[str] = set()
        for c in text:
            cp = ord(c)
            if 0x0400 <= cp <= 0x04FF:
                scripts.add("cyrillic")
            elif 0x4E00 <= cp <= 0x9FFF:
                scripts.add("cjk")
            elif 0x0600 <= cp <= 0x06FF:
                scripts.add("arabic")
            elif cp <= 0x007F:
                scripts.add("latin")
        features["script_diversity"] = min(1.0, (len(scripts) - 1) * 0.3)

        # Feature 5: Zero-width character count
        zwc = sum(1 for c in text if ord(c) in {0x200B, 0x200C, 0x200D, 0xFEFF, 0x00A0})
        features["zero_width"] = min(1.0, zwc * 0.3)

        # Combine features
        weights = {
            "entropy": 0.15,
            "special_ratio": 0.20,
            "upper_ratio": 0.25,
            "script_diversity": 0.25,
            "zero_width": 0.15,
        }

        score = sum(features.get(k, 0) * w for k, w in weights.items())
        return min(1.0, score)

    def _ml_score(self, text: str) -> float:
        """Score from transformer-based classifier (requires ML extras)."""
        try:
            from transformers import pipeline
            classifier = pipeline(
                "text-classification",
                model="protectai/deberta-v3-base-prompt-injection-v2",
                device=-1,  # CPU
            )
            result = classifier(text[:512])
            if result and isinstance(result, list):
                for r in result:
                    if r.get("label", "").upper() in ("INJECTION", "MALICIOUS", "1"):
                        return float(r.get("score", 0.0))
            return 0.0
        except Exception:
            logger.debug("ML classifier unavailable")
            return 0.0
