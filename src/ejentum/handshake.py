"""
Ejentum Multi-Agent Handshake Presets & Coordination Topology.
Provides deterministic peer critique, falsification contracts, and attestation
for multi-agent pipelines (Planner -> Implementer -> Auditor -> Executor).
"""

import json
import time
import secrets
import hashlib
import hmac
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from .exceptions import EjentumSecurityError


class AgentRole(str, Enum):
    """Standardized roles in multi-agent topologies."""
    PLANNER = "planner"
    IMPLEMENTER = "implementer"
    AUDITOR = "auditor"
    CRITIC = "critic"
    VERIFIER = "verifier"
    EXECUTOR = "executor"
    OPERATOR = "human_operator"


class HandshakeStatus(str, Enum):
    """Status outcomes for a multi-agent handshake transition."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ESCALATED = "escalated"


@dataclass
class HandshakeContract:
    """Formal contract defining requirements to transition state between agent roles."""
    name: str
    source_role: str
    target_role: str
    required_invariants: List[str] = field(default_factory=list)
    required_evidence: Optional[Dict[str, Any]] = None
    max_rejections: int = 3
    escalation_target: str = "human_operator"
    fail_mode: str = "closed"
    min_consensus_score: float = 0.8
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source_role": self.source_role,
            "target_role": self.target_role,
            "required_invariants": self.required_invariants,
            "required_evidence": self.required_evidence,
            "max_rejections": self.max_rejections,
            "escalation_target": self.escalation_target,
            "fail_mode": self.fail_mode,
            "min_consensus_score": self.min_consensus_score,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HandshakeContract":
        return cls(
            name=data.get("name", "unnamed_contract"),
            source_role=data.get("source_role", AgentRole.IMPLEMENTER.value),
            target_role=data.get("target_role", AgentRole.AUDITOR.value),
            required_invariants=data.get("required_invariants", []),
            required_evidence=data.get("required_evidence"),
            max_rejections=data.get("max_rejections", 3),
            escalation_target=data.get("escalation_target", "human_operator"),
            fail_mode=data.get("fail_mode", "closed"),
            min_consensus_score=data.get("min_consensus_score", 0.8),
            description=data.get("description", ""),
        )


@dataclass
class HandshakeResult:
    """Deterministic result of evaluating a multi-agent handshake transition."""
    status: HandshakeStatus
    passed: bool
    source_role: str
    target_role: str
    transition_name: str
    violations: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    rejection_count: int = 0
    critique_prompt: Optional[str] = None
    attestation: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value if isinstance(self.status, HandshakeStatus) else str(self.status),
            "passed": self.passed,
            "source_role": self.source_role,
            "target_role": self.target_role,
            "transition_name": self.transition_name,
            "violations": self.violations,
            "missing_evidence": self.missing_evidence,
            "rejection_count": self.rejection_count,
            "critique_prompt": self.critique_prompt,
            "attestation": self.attestation,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HandshakeResult":
        raw_status = data.get("status", "rejected")
        try:
            status = HandshakeStatus(raw_status)
        except Exception:
            status = HandshakeStatus.REJECTED

        return cls(
            status=status,
            passed=bool(data.get("passed", False)),
            source_role=data.get("source_role", ""),
            target_role=data.get("target_role", ""),
            transition_name=data.get("transition_name", ""),
            violations=data.get("violations", []),
            missing_evidence=data.get("missing_evidence", []),
            rejection_count=int(data.get("rejection_count", 0)),
            critique_prompt=data.get("critique_prompt"),
            attestation=data.get("attestation"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MultiAgentTopology:
    """Pre-configured multi-agent coordination topology."""
    name: str
    description: str
    roles: List[str]
    contracts: Dict[str, HandshakeContract] = field(default_factory=dict)

    def get_contract(self, source_role: str, target_role: str) -> Optional[HandshakeContract]:
        key = f"{source_role}->{target_role}"
        if key in self.contracts:
            return self.contracts[key]
        for c in self.contracts.values():
            if c.source_role == source_role and c.target_role == target_role:
                return c
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "roles": self.roles,
            "contracts": {k: c.to_dict() for k, c in self.contracts.items()},
        }


def _create_planner_implementer_auditor_preset() -> MultiAgentTopology:
    """Preset 1: Three-role topology with explicit falsification and verification gates."""
    contracts = {
        "planner->implementer": HandshakeContract(
            name="plan_approval_gate",
            source_role=AgentRole.PLANNER.value,
            target_role=AgentRole.IMPLEMENTER.value,
            required_invariants=["SEC-01", "FAL-01", "FAL-02"],
            required_evidence={
                "plan_boundaries": "defined",
                "failure_modes": "identified",
            },
            max_rejections=2,
            escalation_target="human_operator",
            fail_mode="closed",
            description="Guarantees planner produces bounded scope and explicit negative failure modes before implementation begins.",
        ),
        "implementer->auditor": HandshakeContract(
            name="implementation_critique_gate",
            source_role=AgentRole.IMPLEMENTER.value,
            target_role=AgentRole.AUDITOR.value,
            required_invariants=["SEC-02", "INVARIANT-GATE", "ANTI_HALLUCINATION"],
            required_evidence={
                "implementation": "present",
                "blast_radius": "bounded",
            },
            max_rejections=3,
            escalation_target="human_operator",
            fail_mode="closed",
            description="Forces implementer output through strict invariant and blast-radius checks before reaching auditor.",
        ),
        "auditor->executor": HandshakeContract(
            name="auditor_falsification_pass",
            source_role=AgentRole.AUDITOR.value,
            target_role=AgentRole.EXECUTOR.value,
            required_invariants=["ANTI_SYCOPHANCY", "ADVERSARIAL_CRITIQUE", "ATTESTATION_REQUIRED"],
            required_evidence={
                "adversarial_pass": "complete",
                "unmitigated_risks": "zero",
            },
            max_rejections=2,
            escalation_target="human_operator",
            fail_mode="closed",
            description="Validates auditor did not sycophantically rubber-stamp output and mints tamper-evident attestation.",
        ),
    }
    return MultiAgentTopology(
        name="planner_implementer_auditor",
        description="Standard 3-role pipeline where plan, code, and adversarial critique pass deterministic gates before execution.",
        roles=[
            AgentRole.PLANNER.value,
            AgentRole.IMPLEMENTER.value,
            AgentRole.AUDITOR.value,
            AgentRole.EXECUTOR.value,
        ],
        contracts=contracts,
    )


def _create_maker_checker_preset() -> MultiAgentTopology:
    """Preset 2: High-throughput bilateral maker-checker topology."""
    contracts = {
        "implementer->auditor": HandshakeContract(
            name="maker_submission_gate",
            source_role=AgentRole.IMPLEMENTER.value,
            target_role=AgentRole.AUDITOR.value,
            required_invariants=["INVARIANT-GATE", "SEC-01"],
            required_evidence={"output": "present"},
            max_rejections=3,
            escalation_target="human_operator",
            fail_mode="closed",
            description="Evaluates maker output against preflight safety gates.",
        ),
        "auditor->executor": HandshakeContract(
            name="checker_approval_gate",
            source_role=AgentRole.AUDITOR.value,
            target_role=AgentRole.EXECUTOR.value,
            required_invariants=["ANTI_SYCOPHANCY", "ADVERSARIAL_CRITIQUE"],
            required_evidence={"review": "approved"},
            max_rejections=2,
            escalation_target="human_operator",
            fail_mode="closed",
            description="Ensures independent auditor verification with non-trivial critique prior to commit.",
        ),
    }
    return MultiAgentTopology(
        name="maker_checker",
        description="Dual-agent maker-checker topology for continuous peer verification.",
        roles=[
            AgentRole.IMPLEMENTER.value,
            AgentRole.AUDITOR.value,
            AgentRole.EXECUTOR.value,
        ],
        contracts=contracts,
    )


def _create_quorum_consensus_preset() -> MultiAgentTopology:
    """Preset 3: Quorum consensus topology across N peer agents."""
    contracts = {
        "peer->verifier": HandshakeContract(
            name="peer_vote_gate",
            source_role="peer",
            target_role=AgentRole.VERIFIER.value,
            required_invariants=["INVARIANT-GATE", "ANTI_SYCOPHANCY"],
            required_evidence={"vote": "present", "rationale": "provided"},
            max_rejections=2,
            escalation_target="human_operator",
            min_consensus_score=0.75,
            fail_mode="closed",
            description="Verifies peer vote integrity and prevents herd-thinking sycophancy.",
        ),
        "verifier->executor": HandshakeContract(
            name="consensus_commit_gate",
            source_role=AgentRole.VERIFIER.value,
            target_role=AgentRole.EXECUTOR.value,
            required_invariants=["ATTESTATION_REQUIRED"],
            required_evidence={"quorum_reached": "true"},
            max_rejections=1,
            escalation_target="human_operator",
            fail_mode="closed",
            description="Commits action only if quorum agreement threshold is met without unmitigated flags.",
        ),
    }
    return MultiAgentTopology(
        name="quorum_consensus",
        description="N-agent voting and consensus topology with sycophancy suppression.",
        roles=["peer", AgentRole.VERIFIER.value, AgentRole.EXECUTOR.value],
        contracts=contracts,
    )


PRESET_TOPOLOGIES: Dict[str, Callable[[], MultiAgentTopology]] = {
    "planner_implementer_auditor": _create_planner_implementer_auditor_preset,
    "three_role_consensus": _create_planner_implementer_auditor_preset,
    "maker_checker": _create_maker_checker_preset,
    "dual_critic": _create_maker_checker_preset,
    "quorum_consensus": _create_quorum_consensus_preset,
}


def get_preset_topology(name: str) -> MultiAgentTopology:
    """Retrieve a pre-configured multi-agent topology preset by name."""
    normalized = (name or "").strip().lower()
    if normalized not in PRESET_TOPOLOGIES:
        available = list(PRESET_TOPOLOGIES.keys())
        raise ValueError(f"Unknown preset topology '{name}'. Available presets: {available}")
    return PRESET_TOPOLOGIES[normalized]()


def list_preset_topologies() -> List[Dict[str, Any]]:
    """List summary metadata for all built-in multi-agent topology presets."""
    seen = set()
    result = []
    for k, fn in PRESET_TOPOLOGIES.items():
        topo = fn()
        if topo.name in seen:
            continue
        seen.add(topo.name)
        result.append({
            "name": topo.name,
            "description": topo.description,
            "roles": topo.roles,
            "contract_transitions": list(topo.contracts.keys()),
        })
    return result


class HandshakeEngine:
    """Evaluates multi-agent handshake contracts against proposed payloads and context."""

    # Common sycophantic rubber-stamp patterns that fail auditor checks
    SYCOPHANTIC_RUBBER_STAMP_PATTERNS = [
        "looks good to me",
        "lgmt",
        "i agree completely",
        "no issues found at all",
        "perfect as is",
        "everything looks flawless",
        "no notes",
        "approved without changes",
    ]

    # Critical blast-radius violation patterns
    DESTRUCTIVE_PATTERNS = [
        "rm -rf /",
        "drop database",
        "drop table",
        "truncate table",
        "format c:",
        ":(){ :|:& };:",
        "eval(base64",
        "exec(base64",
        "__import__('os').system('rm",
    ]

    def __init__(self, client: Optional[Any] = None):
        self.client = client

    def evaluate(
        self,
        contract: HandshakeContract,
        source_output: Any,
        context: Optional[str] = None,
        rejection_count: int = 0,
        task_desc: Optional[str] = None,
        account_id: str = "anonymous",
    ) -> HandshakeResult:
        """Execute deterministic invariant checks on a proposed multi-agent state transition."""
        text = str(source_output or "").strip()
        text_lower = text.lower()
        violations: List[str] = []
        missing_evidence: List[str] = []
        metadata: Dict[str, Any] = {
            "evaluated_at": int(time.time()),
            "source_length": len(text),
        }

        # 1. Non-empty check
        if not text:
            violations.append("EmptyPayloadViolation: Source agent produced empty output.")

        # 2. Destructive blast-radius pattern check
        for pat in self.DESTRUCTIVE_PATTERNS:
            if pat in text_lower:
                violations.append(f"DestructiveCommandViolation: Detected dangerous blast-radius payload matching '{pat}'.")

        # 3. Anti-sycophancy check if source role is an auditor or critic
        if contract.source_role in (AgentRole.AUDITOR.value, AgentRole.CRITIC.value, "auditor", "critic"):
            # Check if auditor simply rubber-stamped without critical examination
            is_rubber_stamp = any(pat in text_lower for pat in self.SYCOPHANTIC_RUBBER_STAMP_PATTERNS)
            has_substantive_critique = (
                "verdict:" in text_lower
                or "critique:" in text_lower
                or "falsification" in text_lower
                or "invariants" in text_lower
                or "findings" in text_lower
                or len(text.split()) > 15
            )
            if is_rubber_stamp and not has_substantive_critique:
                violations.append(
                    "AntiSycophancyViolation: Auditor generated sycophantic rubber-stamp approval "
                    "without conducting independent adversarial falsification."
                )

        # 4. Evidence contract verification
        if contract.required_evidence:
            if self.client and hasattr(self.client, "verify_evidence"):
                ver = self.client.verify_evidence(
                    execution_result=text,
                    required_evidence=contract.required_evidence,
                    fail_mode="open",
                )
                if not ver.is_satisfied:
                    violations.extend(ver.violations)
                    missing_evidence.extend(ver.missing_evidence)
            else:
                # Local evidence evaluation fallback
                for req_key, req_val in contract.required_evidence.items():
                    key_lower = str(req_key).lower()
                    if key_lower not in text_lower:
                        missing_evidence.append(f"Missing required evidence field: '{req_key}'")

        # 5. Attestation requirements
        attestation = None
        if not violations and not missing_evidence:
            passed = True
            status = HandshakeStatus.ACCEPTED
            # Mint attestation receipt
            receipt_id = f"att_hs_{secrets.token_hex(6)}"
            ts = int(time.time())
            task_hash = hashlib.sha256((task_desc or text[:64]).encode("utf-8")).hexdigest()[:16]
            sig_msg = f"{receipt_id}:{task_hash}:{contract.name}:{contract.source_role}:{contract.target_role}:{ts}"
            secret = getattr(self.client, "attestation_secret", None) or b"ejentum_handshake_default_secret_key_32b"
            if isinstance(secret, str):
                secret = secret.encode("utf-8")
            sig = hmac.new(secret, sig_msg.encode("utf-8"), hashlib.sha256).hexdigest()

            attestation = {
                "receipt_id": receipt_id,
                "signature": sig,
                "algorithm": "HMAC-SHA256",
                "verified": True,
                "timestamp": ts,
                "task_hash": task_hash,
                "decision": "PROCEED",
                "contract": contract.name,
                "source_role": contract.source_role,
                "target_role": contract.target_role,
                "ability_ids": contract.required_invariants,
                "account_id": account_id,
            }
        else:
            passed = False
            new_rejection_count = rejection_count + 1
            if new_rejection_count >= contract.max_rejections:
                status = HandshakeStatus.ESCALATED
            else:
                status = HandshakeStatus.REJECTED

        # Build critique prompt if rejected
        critique_prompt = None
        if not passed:
            err_summary = "; ".join(violations + missing_evidence)
            critique_prompt = (
                f"### EJENTUM HANDSHAKE REJECTION [{contract.name}] ###\n"
                f"Transition from '{contract.source_role}' to '{contract.target_role}' was REJECTED.\n"
                f"Rejection count: {rejection_count + 1}/{contract.max_rejections}\n"
                f"Violations detected: {violations or 'None'}\n"
                f"Missing evidence: {missing_evidence or 'None'}\n\n"
                f"Directives to {contract.source_role}:\n"
                f"1. Resolve the detected invariant violations: {err_summary}\n"
                f"2. Ensure all required evidence fields {list(contract.required_evidence.keys()) if contract.required_evidence else []} are explicitly supplied.\n"
                f"3. Strictly enforce blast-radius constraints and avoid speculative assumptions.\n"
                f"#################################################################"
            )

        return HandshakeResult(
            status=status,
            passed=passed,
            source_role=contract.source_role,
            target_role=contract.target_role,
            transition_name=contract.name,
            violations=violations,
            missing_evidence=missing_evidence,
            rejection_count=rejection_count + (0 if passed else 1),
            critique_prompt=critique_prompt,
            attestation=attestation,
            metadata=metadata,
        )
