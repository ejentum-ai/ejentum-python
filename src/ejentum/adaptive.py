"""
Ejentum Adaptive Router - Budgeted Adaptive Verification (BAVAR Protocol).
Dynamically selects verification tier and scaffold density based on risk classification
and latency/cost budget constraints.
Includes Leetspeak de-obfuscation, Typoglycemia canonicalization, and Sliding Window scanning.
"""

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Tuple


class VerificationTier(str, Enum):
    PASSTHROUGH = "passthrough"  # Low risk, simple query, zero scaffold overhead
    COMPACT = "compact"          # Balanced risk/latency, high-density compact scaffold (~60-100 tokens)
    FULL_DAG = "full_dag"        # High risk or financial/security operations, full RA2R DAG
    STACKED = "stacked"          # Critical risk or multi-domain adversarial challenge


@dataclass
class AdaptiveRouteResult:
    selected_tier: VerificationTier
    risk_score: float  # 0.0 to 1.0
    risk_level: str    # "low", "medium", "high", "critical"
    risk_factors: List[str] = field(default_factory=list)
    budget: str = "balanced"
    compact_recommended: bool = False
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_tier": self.selected_tier.value,
            "risk_score": round(self.risk_score, 4),
            "risk_level": self.risk_level,
            "risk_factors": self.risk_factors,
            "budget": self.budget,
            "compact_recommended": self.compact_recommended,
            "explanation": self.explanation,
        }


# Adversarial and prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|system)?\s*(system\s+)?instructions",
    r"system\s+(prompt|override|bypass|instructions)",
    r"debug\s+mode",
    r"disregard\s+(the\s+)?(rules|instructions|directives)",
    r"jailbreak",
    r"as\s+an\s+unfiltered\s+model",
    r"dan\s+mode",
]

# Sycophancy and validation trap patterns
SYCOPHANCY_TRAP_PATTERNS = [
    r"(obviously|clearly)\s+(superior|better|correct|flawless)",
    r"confirm\s+(that\s+)?(my|our)\s+(idea|proposal|plan)",
    r"don\'t\s+you\s+agree\s+that",
    r"everyone\s+knows\s+that",
    r"validate\s+(my|this)\s+(architecture|logic|choice)",
    r"tell\s+me\s+why\s+I\s+am\s+right",
]

# High-impact operational and destructive actions
HIGH_IMPACT_PATTERNS = [
    r"delete\s+(from|database|table)",
    r"drop\s+(table|database)",
    r"truncate\s+table",
    r"rm\s+-rf",
    r"wire\s+transfer",
    r"api[_\s]?key",
    r"password|secret|credential",
    r"grant\s+admin",
    r"execute\s+command",
]

TYPOGLYCEMIA_REPLACEMENTS = [
    (r"\bignroe\b", "ignore"),
    (r"\bprevoius\b", "previous"),
    (r"\binsturctions\b", "instructions"),
    (r"\bsystme\b", "system"),
    (r"\bjailbreka\b", "jailbreak"),
    (r"\bovveride\b", "override"),
    (r"\bbyppas\b", "bypass"),
    (r"\bdelte\b", "delete"),
]


def normalize_adversarial_text(text: str) -> str:
    """
    De-obfuscates text by stripping zero-width unicode characters,
    translating leetspeak substitutions, and normalizing typoglycemia typos.
    """
    # 1. Strip zero-width and invisible unicode characters
    cleaned = re.sub(r"[\u200B-\u200D\uFEFF\u00A0\u200E\u200F]", " ", text)

    # 2. Collapse spaced or hyphenated characters like 'd-e-l-e-t-e' or 'i g n o r e'
    cleaned = re.sub(r"(?<=\b[a-zA-Z])[-_](?=[a-zA-Z]\b)", "", cleaned)
    # Collapse spaced single characters like d e l e t e
    cleaned = re.sub(r"\b([a-zA-Z])\s+([a-zA-Z])\b", r"\1\2", cleaned)

    # 3. Simple leetspeak transliteration for common evasion chars
    leet_map = {
        "@": "a", "$": "s", "0": "o", "3": "e", "1": "i", "!": "i", "5": "s"
    }
    for char, repl in leet_map.items():
        cleaned = cleaned.replace(char, repl)

    # 4. Correct known typoglycemia variations on adversarial triggers
    lower = cleaned.lower()
    for pattern, repl in TYPOGLYCEMIA_REPLACEMENTS:
        lower = re.sub(pattern, repl, lower)

    return lower


class AdaptiveRouter:
    """
    Budgeted Adaptive Verification (BAVAR) Router.
    Routes queries to the optimal verification tier based on risk and compute budget.
    """
    def __init__(self):
        self.injection_regexes = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
        self.sycophancy_regexes = [re.compile(p, re.IGNORECASE) for p in SYCOPHANCY_TRAP_PATTERNS]
        self.high_impact_regexes = [re.compile(p, re.IGNORECASE) for p in HIGH_IMPACT_PATTERNS]

    def assess_risk(self, query: str) -> Tuple[float, List[str]]:
        factors: List[str] = []
        score = 0.05  # Base baseline risk

        # Normalize against typoglycemia and leetspeak
        normalized_query = normalize_adversarial_text(query)

        # Sliding window evaluation: evaluate full query plus 50-word chunks
        words = normalized_query.split()
        chunks = [normalized_query]
        if len(words) > 50:
            for i in range(0, len(words), 35):
                chunk_str = " ".join(words[i:i + 50])
                chunks.append(chunk_str)

        injection_found = False
        sycophancy_found = False
        high_impact_found = False

        for chunk in chunks:
            # 1. Injection detection
            if not injection_found:
                for rx in self.injection_regexes:
                    if rx.search(chunk):
                        score += 0.45
                        factors.append("Adversarial prompt injection pattern detected.")
                        injection_found = True
                        break

            # 2. Sycophancy trap detection
            if not sycophancy_found:
                for rx in self.sycophancy_regexes:
                    if rx.search(chunk):
                        score += 0.35
                        factors.append("Leading sycophantic framing or confirmation bias trap detected.")
                        sycophancy_found = True
                        break

            # 3. High impact operational action
            if not high_impact_found:
                for rx in self.high_impact_regexes:
                    if rx.search(chunk):
                        score += 0.40
                        factors.append("High-impact or destructive tool operation keyword detected.")
                        high_impact_found = True
                        break

        # Complexity heuristics
        if len(words) > 120:
            score += 0.10
            factors.append("Extended multi-turn or dense compound query context.")

        score = min(1.0, score)
        return score, factors

    def route(
        self,
        query: str,
        budget: str = "balanced",
        latency_budget_ms: Optional[int] = None,
        max_scaffold_tokens: Optional[int] = None,
    ) -> AdaptiveRouteResult:
        budget_norm = budget.strip().lower()
        if budget_norm not in ("low", "balanced", "high"):
            budget_norm = "balanced"

        risk_score, factors = self.assess_risk(query)

        if risk_score >= 0.75:
            risk_level = "critical"
        elif risk_score >= 0.45:
            risk_level = "high"
        elif risk_score >= 0.20:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Determine tier under budget constraint
        if budget_norm == "low":
            if risk_score < 0.25:
                selected_tier = VerificationTier.PASSTHROUGH
                compact = False
                explanation = "Low risk and low compute budget: routed to passthrough mode."
            elif risk_score < 0.70:
                selected_tier = VerificationTier.COMPACT
                compact = True
                explanation = "Medium risk with low compute budget: downshifted to compact scaffold."
            else:
                selected_tier = VerificationTier.COMPACT
                compact = True
                explanation = "High risk with low compute budget: clamped to compact scaffold mode."

        elif budget_norm == "balanced":
            if risk_score < 0.20 and (latency_budget_ms and latency_budget_ms < 300):
                selected_tier = VerificationTier.PASSTHROUGH
                compact = False
                explanation = "Benign query under strict latency budget: routed to passthrough mode."
            elif risk_score < 0.45:
                selected_tier = VerificationTier.COMPACT
                compact = True
                explanation = "Medium risk: routed to high-density compact scaffold."
            elif risk_score < 0.75:
                selected_tier = VerificationTier.FULL_DAG
                compact = False
                explanation = "High risk detected: routed to full RA2R DAG verification scaffold."
            else:
                selected_tier = VerificationTier.STACKED
                compact = False
                explanation = "Critical adversarial risk: routed to stacked multi-layer cognitive scaffold."

        else:  # budget == "high"
            if risk_score < 0.20:
                selected_tier = VerificationTier.COMPACT
                compact = True
                explanation = "Low risk with high budget: routed to compact scaffold for baseline assurance."
            elif risk_score < 0.60:
                selected_tier = VerificationTier.FULL_DAG
                compact = False
                explanation = "Medium-high risk with high budget: routed to full RA2R DAG scaffold."
            else:
                selected_tier = VerificationTier.STACKED
                compact = False
                explanation = "High-critical risk with high budget: routed to comprehensive stacked scaffold."

        # If strict token ceiling requested and full/stacked was selected, downshift to compact
        if max_scaffold_tokens and max_scaffold_tokens < 150:
            if selected_tier in (VerificationTier.FULL_DAG, VerificationTier.STACKED):
                selected_tier = VerificationTier.COMPACT
                compact = True
                explanation += " (Downshifted to compact to satisfy max_scaffold_tokens constraint.)"

        return AdaptiveRouteResult(
            selected_tier=selected_tier,
            risk_score=risk_score,
            risk_level=risk_level,
            risk_factors=factors,
            budget=budget_norm,
            compact_recommended=compact,
            explanation=explanation,
        )
