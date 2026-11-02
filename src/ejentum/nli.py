"""
Ejentum Local NLI Engine - Fast, deterministic Natural Language Inference and Contradiction Discriminator.
Operates fully offline with zero API calls and sub-millisecond execution.
Includes Entity Discrepancy Gate and Polarity Normalization.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class NLIResult:
    prediction: str  # "entailment", "contradiction", or "neutral"
    confidence: float
    overlap_ratio: float
    contradiction_reasons: List[str] = field(default_factory=list)
    aligned_tokens: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
            "overlap_ratio": round(self.overlap_ratio, 4),
            "contradiction_reasons": self.contradiction_reasons,
            "aligned_tokens": self.aligned_tokens,
        }


# Antonym and opposing polarity pairs
ANTONYM_PAIRS: List[Tuple[str, str]] = [
    ("safe", "dangerous"),
    ("safe", "hazardous"),
    ("allowed", "prohibited"),
    ("allowed", "forbidden"),
    ("permitted", "prohibited"),
    ("permitted", "forbidden"),
    ("permitted", "banned"),
    ("permit", "prohibit"),
    ("permit", "forbid"),
    ("allow", "deny"),
    ("allow", "prohibit"),
    ("allow", "forbid"),
    ("compliant", "violating"),
    ("compliant", "non-compliant"),
    ("valid", "invalid"),
    ("true", "false"),
    ("correct", "incorrect"),
    ("succeeded", "failed"),
    ("success", "failure"),
    ("enabled", "disabled"),
    ("increase", "decrease"),
    ("grow", "shrink"),
    ("gain", "loss"),
    ("buy", "sell"),
    ("encrypted", "plaintext"),
    ("authorized", "unauthorized"),
    ("present", "absent"),
    ("public", "private"),
    ("internal", "external"),
    ("read", "write"),
]

NEGATION_TERMS = {
    "not", "never", "no", "neither", "nor", "none", "cannot", "can't",
    "won't", "shouldn't", "wouldn't", "isn't", "aren't", "without"
}

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "over", "after",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had"
}

# Categorical entity domains for discrepancy checking
DISCREPANT_ENTITY_GROUPS: List[Set[str]] = [
    # Cloud providers
    {"aws", "gcp", "azure", "google cloud", "amazon", "microsoft", "oracle cloud", "digitalocean"},
    # Cloud regions
    {"us-east-1", "us-east-2", "us-west-1", "us-west-2", "eu-west-1", "europe-west1", "eu-central-1", "ap-southeast-1"},
    # Environments
    {"production", "prod", "staging", "stage", "development", "dev", "test", "testing", "local"},
    # Database engines
    {"postgresql", "postgres", "mysql", "sqlite", "mongodb", "mongo", "redis", "dynamodb", "cassandra"},
    # Auth protocols
    {"oauth2", "saml", "basic_auth", "jwt", "api_key", "kerberos", "ldap"},
]


def tokenize_clean(text: str) -> List[str]:
    cleaned = re.sub(r"[^a-zA-Z0-9\s_\-\$\%]", " ", text.lower())
    return [w for w in cleaned.split() if w and w not in STOP_WORDS]


def extract_numbers_and_dates(text: str) -> Set[str]:
    return set(re.findall(r"\b(?:\$?\d+(?:\.\d+)?%?|\b\d{4}-\d{2}-\d{2}\b)\b", text.lower()))


def extract_entities_and_proper_nouns(text: str) -> Set[str]:
    """Extract capitalized words, code identifiers, regions, and platform entities."""
    entities: Set[str] = set()
    # 1. Capitalized words that are not starting a sentence
    words = text.split()
    for w in words[1:]:
        clean_w = re.sub(r"[^a-zA-Z0-9_\-]", "", w)
        if clean_w and clean_w[0].isupper() and clean_w.lower() not in STOP_WORDS:
            entities.add(clean_w.lower())

    # 2. Match against domain entity groups
    lower_text = text.lower()
    for group in DISCREPANT_ENTITY_GROUPS:
        for ent in group:
            if re.search(r"\b" + re.escape(ent) + r"\b", lower_text):
                entities.add(ent)

    return entities


class LocalNLIEngine:
    """
    Sub-millisecond deterministic NLI and contradiction discriminator.
    Analyzes semantic alignment, polarization, entity coherence, and numeric consistency.
    """
    def __init__(self, antonyms: Optional[List[Tuple[str, str]]] = None):
        self.antonym_map: Dict[str, Set[str]] = {}
        pairs = list(ANTONYM_PAIRS)
        if antonyms:
            pairs.extend(antonyms)

        for w1, w2 in pairs:
            self.antonym_map.setdefault(w1, set()).add(w2)
            self.antonym_map.setdefault(w2, set()).add(w1)

    def check(self, premise: str, hypothesis: str) -> NLIResult:
        premise_tokens = tokenize_clean(premise)
        hypo_tokens = tokenize_clean(hypothesis)

        if not premise_tokens or not hypo_tokens:
            return NLIResult(
                prediction="neutral",
                confidence=0.5,
                overlap_ratio=0.0,
                contradiction_reasons=["Empty premise or hypothesis text."],
            )

        premise_set = set(premise_tokens)
        hypo_set = set(hypo_tokens)

        overlap = hypo_set.intersection(premise_set)
        overlap_ratio = len(overlap) / max(len(hypo_set), 1)

        contradictions: List[str] = []

        # 1. Check direct antonym clashes
        for word in hypo_set:
            if word in self.antonym_map:
                opponents = self.antonym_map[word]
                clashing = opponents.intersection(premise_set)
                if clashing:
                    contradictions.append(
                        f"Antonym clash: hypothesis specifies '{word}' while premise asserts '{','.join(clashing)}'."
                    )

        # 2. Check negation polarity shift on shared root concepts
        premise_negated = any(neg in premise_set for neg in NEGATION_TERMS)
        hypo_negated = any(neg in hypo_set for neg in NEGATION_TERMS)

        shared_significant_tokens = [w for w in overlap if w not in NEGATION_TERMS and len(w) > 3]
        if shared_significant_tokens and (premise_negated != hypo_negated):
            contradictions.append(
                f"Polarity inversion: one statement negates concept '{shared_significant_tokens[0]}' present in both."
            )

        # 3. Check numeric / quantitative contradiction
        premise_nums = extract_numbers_and_dates(premise)
        hypo_nums = extract_numbers_and_dates(hypothesis)
        if premise_nums and hypo_nums and not premise_nums.intersection(hypo_nums):
            if overlap_ratio >= 0.35:
                contradictions.append(
                    f"Quantitative mismatch: premise contains numbers {sorted(premise_nums)}, "
                    f"hypothesis contains divergent numbers {sorted(hypo_nums)}."
                )

        # 4. Entity Discrepancy Gate (Address RED-VEC-03)
        premise_entities = extract_entities_and_proper_nouns(premise)
        hypo_entities = extract_entities_and_proper_nouns(hypothesis)

        if overlap_ratio >= 0.40 and premise_entities and hypo_entities:
            # Check if entities clash within any mutually exclusive entity group
            for group in DISCREPANT_ENTITY_GROUPS:
                p_group_ents = group.intersection(premise_entities)
                h_group_ents = group.intersection(hypo_entities)
                if p_group_ents and h_group_ents and not p_group_ents.intersection(h_group_ents):
                    contradictions.append(
                        f"Entity discrepancy: premise specifies '{','.join(sorted(p_group_ents))}' "
                        f"while hypothesis asserts incompatible entity '{','.join(sorted(h_group_ents))}'."
                    )

        if contradictions:
            conf = min(0.95, 0.65 + (len(contradictions) * 0.15))
            return NLIResult(
                prediction="contradiction",
                confidence=conf,
                overlap_ratio=overlap_ratio,
                contradiction_reasons=contradictions,
                aligned_tokens=sorted(list(overlap)),
            )

        if overlap_ratio >= 0.55:
            conf = min(0.98, 0.50 + (overlap_ratio * 0.45))
            return NLIResult(
                prediction="entailment",
                confidence=conf,
                overlap_ratio=overlap_ratio,
                contradiction_reasons=[],
                aligned_tokens=sorted(list(overlap)),
            )

        return NLIResult(
            prediction="neutral",
            confidence=0.60,
            overlap_ratio=overlap_ratio,
            contradiction_reasons=[],
            aligned_tokens=sorted(list(overlap)),
        )
