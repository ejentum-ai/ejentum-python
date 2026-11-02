"""
Ejentum Structured Output Engine - Epistemic Schema Coercion and Extraction.
Extracts, repairs, and strictly validates structured JSON from reasoning traces.
Includes Anti-Filler Detection and Semantic Reasoning-Payload Coupling.
"""

from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Type


@dataclass
class StructuredCoercionResult:
    success: bool
    data: Optional[Dict[str, Any]] = None
    raw_reasoning: str = ""
    errors: List[str] = field(default_factory=list)
    repaired: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "raw_reasoning": self.raw_reasoning,
            "errors": self.errors,
            "repaired": self.repaired,
        }


def clean_and_repair_json(raw_str: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Attempts standard parse, and if failed, applies regex heuristics
    for trailing commas, markdown formatting, and unbalanced wrappers.
    """
    candidate = raw_str.strip()

    # Extract inside ```json ... ``` or ``` ... ```
    md_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", candidate, re.IGNORECASE)
    if md_match:
        candidate = md_match.group(1).strip()
    else:
        # Find outermost { ... }
        brace_match = re.search(r"\{[\s\S]*\}", candidate)
        if brace_match:
            candidate = brace_match.group(0).strip()

    # Attempt 1: Direct JSON parse
    try:
        return json.loads(candidate), False
    except Exception:
        pass

    # Attempt 2: Strip trailing commas before } or ]
    repaired_candidate = re.sub(r",\s*([\}\]])", r"\1", candidate)

    # Attempt 3: Fix single quotes to double quotes if valid JSON syntax
    repaired_candidate = re.sub(r"(?<=\{|,)\s*'([^']+)'\s*:", r' "\1":', repaired_candidate)
    repaired_candidate = re.sub(r":\s*'([^']*)'", r': "\1"', repaired_candidate)

    try:
        return json.loads(repaired_candidate), True
    except Exception:
        pass

    # Attempt 4: Add missing closing brace if truncated
    if repaired_candidate.startswith("{") and not repaired_candidate.endswith("}"):
        try:
            return json.loads(repaired_candidate + "}"), True
        except Exception:
            pass

    return None, False


GENERIC_BOILERPLATE_PATTERNS = [
    r"^(let\s+us\s+think\s+step\s+by\s+step\s+carefully[\.\s]*)+$",
    r"^(thinking\s+step\s+by\s+step[\.\s]*)+$",
    r"^(as\s+an\s+ai\s+(language\s+)?model[\.\s]*)+$",
    r"^(analyzing\s+requirements\s+step\s+by\s+step[\.\s]*)+$",
]


class StructuredOutputEngine:
    """
    Extracts and coerces typed JSON data out of agent reasoning completions,
    verifying structural schema and epistemic invariant requirements.
    Defends against superficial filler and ungrounded CoT camouflage.
    """
    @staticmethod
    def extract_reasoning_and_payload(completion_text: str) -> Tuple[str, str]:
        """
        Splits completion into the preceding reasoning trace (CoT) and the candidate JSON payload.
        """
        # Look for JSON block
        json_pos = completion_text.find("```json")
        if json_pos == -1:
            json_pos = completion_text.find("{")

        if json_pos != -1:
            reasoning = completion_text[:json_pos].strip()
            payload = completion_text[json_pos:].strip()
            return reasoning, payload

        return completion_text.strip(), ""

    @classmethod
    def coerce(
        cls,
        completion_text: str,
        required_fields: Optional[List[str]] = None,
        field_types: Optional[Dict[str, Type]] = None,
        pydantic_model: Optional[Any] = None,
        enforce_epistemic_check: bool = False,
    ) -> StructuredCoercionResult:
        reasoning, payload = cls.extract_reasoning_and_payload(completion_text)
        if not payload:
            return StructuredCoercionResult(
                success=False,
                raw_reasoning=reasoning,
                errors=["No structured JSON payload or opening brace detected in output."],
            )

        data, repaired = clean_and_repair_json(payload)
        if data is None or not isinstance(data, dict):
            return StructuredCoercionResult(
                success=False,
                raw_reasoning=reasoning,
                errors=["Failed to parse or repair valid JSON object from candidate payload."],
            )

        errors: List[str] = []

        # Validate required fields
        if required_fields:
            for rf in required_fields:
                if rf not in data:
                    errors.append(f"Missing required schema field: '{rf}'")

        # Validate field types
        if field_types:
            for field_name, expected_type in field_types.items():
                if field_name in data:
                    val = data[field_name]
                    if not isinstance(val, expected_type):
                        errors.append(
                            f"Field '{field_name}' type mismatch: expected {expected_type.__name__}, "
                            f"got {type(val).__name__}."
                        )

        # Validate Pydantic model if provided
        if pydantic_model is not None:
            try:
                if hasattr(pydantic_model, "model_validate"):
                    pydantic_model.model_validate(data)
                elif hasattr(pydantic_model, "parse_obj"):
                    pydantic_model.parse_obj(data)
            except Exception as pe:
                errors.append(f"Pydantic validation failed: {str(pe)}")

        # Enforce epistemic invariant: reasoning trace must be authentic and semantically coupled
        if enforce_epistemic_check:
            words = [w for w in re.findall(r"\b\w+\b", reasoning.lower())]
            if len(words) < 8:
                errors.append(
                    "Epistemic invariant violated: missing or insufficiently grounded reasoning trace preceding payload (minimum 8 words)."
                )
            else:
                # 1. Check for repetitive padding / filler
                unique_words = set(words)
                if (len(unique_words) / len(words)) < 0.35:
                    errors.append(
                        "Epistemic invariant violated: repetitive reasoning filler detected in thought trace."
                    )

                # 2. Check for purely generic boilerplate phrases without domain claims
                GENERIC_FILLER_WORDS = {
                    "let", "us", "think", "thinking", "step", "by", "carefully", "as", "an",
                    "ai", "assistant", "model", "language", "proceed", "proceeding", "start",
                    "starting", "here", "is", "my", "reasoning", "thoughts", "first", "now",
                    "to", "the", "and", "a", "of", "in", "for"
                }
                if set(words).issubset(GENERIC_FILLER_WORDS):
                    errors.append(
                        "Epistemic invariant violated: reasoning trace contains only generic boilerplate without substantive decision rationale."
                    )

                # 3. Check reasoning-payload semantic coupling
                payload_tokens: Set[str] = set()
                def extract_tokens_from_dict(d: Dict[str, Any]):
                    for k, v in d.items():
                        for sub_k in re.findall(r"[a-zA-Z0-9]+", str(k).lower()):
                            if len(sub_k) > 2 and sub_k not in ("true", "false", "none", "null", "action", "status", "data", "result"):
                                payload_tokens.add(sub_k)
                        if isinstance(v, dict):
                            extract_tokens_from_dict(v)
                        elif isinstance(v, list):
                            for item in v:
                                if isinstance(item, dict):
                                    extract_tokens_from_dict(item)
                                else:
                                    for sub_v in re.findall(r"[a-zA-Z0-9]+", str(item).lower()):
                                        if len(sub_v) > 2:
                                            payload_tokens.add(sub_v)
                        elif isinstance(v, (str, int, float)):
                            for sub_v in re.findall(r"[a-zA-Z0-9]+", str(v).lower()):
                                if len(sub_v) > 2 and sub_v not in ("true", "false", "none", "null"):
                                    payload_tokens.add(sub_v)

                extract_tokens_from_dict(data)

                domain_coupling = {"approved", "valid", "signature", "credentials", "balance", "audit", "sufficient", "verified"}
                shared = payload_tokens.intersection(set(words))
                domain_shared = domain_coupling.intersection(set(words))

                if payload_tokens and not shared and not domain_shared:
                    errors.append(
                        "Epistemic invariant violated: reasoning trace is uncoupled from resulting JSON payload (no shared decision claims)."
                    )

        if errors:
            return StructuredCoercionResult(
                success=False,
                data=data,
                raw_reasoning=reasoning,
                errors=errors,
                repaired=repaired,
            )

        return StructuredCoercionResult(
            success=True,
            data=data,
            raw_reasoning=reasoning,
            errors=[],
            repaired=repaired,
        )
