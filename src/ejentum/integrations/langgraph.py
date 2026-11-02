"""
Ejentum LangGraph Integration.
Provides conditional guard nodes and rollback edges that intercept agent execution state.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Union
from ..client import Ejentum
from ..exceptions import EjentumSecurityError
from ..checkpoints import BaseCheckpointStore
from ..handshake import (
    HandshakeContract,
    HandshakeResult,
    HandshakeStatus,
    HandshakeEngine,
    get_preset_topology,
)


class EjentumGuardNode:
    """LangGraph node that guards execution state against cognitive and evidence invariant violations."""

    def __init__(
        self,
        client: Optional[Ejentum] = None,
        task_desc: str = "validate execution state",
        required_evidence: Optional[Union[Dict[str, Any], List[str]]] = None,
        rollback_target: str = "replan",
        success_target: str = "continue",
        fail_mode: str = "closed",
        max_retries: int = 3,
        escalation_target: str = "abort",
        checkpointer: Optional[BaseCheckpointStore] = None,
        thread_id: Optional[str] = None,
        auto_rollback_state: bool = True,
    ):
        if client is not None:
            self.client = client
        elif os.getenv("EJENTUM_API_KEY"):
            self.client = Ejentum()
        else:
            self.client = Ejentum(local=True)
        self.task_desc = task_desc
        self.required_evidence = required_evidence
        self.rollback_target = rollback_target
        self.success_target = success_target
        self.fail_mode = fail_mode
        self.max_retries = max_retries
        self.escalation_target = escalation_target
        self.checkpointer = checkpointer
        self.thread_id = thread_id
        self.auto_rollback_state = auto_rollback_state

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process state and evaluate evidence contracts."""
        last_output = (
            state.get("last_tool_output")
            or state.get("tool_result")
            or state.get("result")
            or state.get("output")
        )
        if last_output is None and "messages" in state and state["messages"]:
            msgs = state["messages"]
            if isinstance(msgs, list) and len(msgs) > 0:
                last_msg = msgs[-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    last_output = last_msg.tool_calls
                elif hasattr(last_msg, "content"):
                    last_output = last_msg.content
                elif isinstance(last_msg, dict):
                    last_output = (
                        last_msg.get("tool_calls")
                        or last_msg.get("content")
                        or last_msg.get("text")
                        or last_msg
                    )
                else:
                    last_output = str(last_msg)
        eval_mode = "open" if self.fail_mode != "abort" else "closed"
        try:
            verification = self.client.verify_evidence(
                execution_result=last_output,
                required_evidence=self.required_evidence,
                fail_mode=eval_mode,
            )
        except EjentumSecurityError:
            if self.fail_mode == "abort":
                raise
            verification = self.client.verify_evidence(
                execution_result=last_output,
                required_evidence=self.required_evidence,
                fail_mode="open",
            )
        thread = self.thread_id or state.get("thread_id") or state.get("session_id") or "default_thread"
        updated_state = dict(state)
        updated_state["ejentum_verification"] = verification.to_dict()
        updated_state["ejentum_is_satisfied"] = verification.is_satisfied

        if not verification.is_satisfied:
            retry_count = int(state.get("ejentum_retry_count", 0)) + 1
            reason = verification.violations or verification.missing_evidence

            if self.checkpointer and self.auto_rollback_state:
                prior = self.checkpointer.rollback(thread_id=thread, steps=1)
                if prior and isinstance(prior.get("state"), dict):
                    updated_state = dict(prior["state"])
                    updated_state["ejentum_verification"] = verification.to_dict()
                    updated_state["ejentum_is_satisfied"] = False
                    updated_state["ejentum_rolled_back_from"] = prior.get("checkpoint_id")

            updated_state["ejentum_retry_count"] = retry_count
            if retry_count >= self.max_retries:
                updated_state["next_step"] = self.escalation_target
                updated_state["rollback_reason"] = (
                    f"Max rollback retries ({self.max_retries}) exceeded: {reason}"
                )
            else:
                updated_state["next_step"] = self.rollback_target
                updated_state["rollback_reason"] = reason
        else:
            updated_state["next_step"] = self.success_target
            updated_state["ejentum_retry_count"] = 0
            if self.checkpointer:
                cp_id = self.checkpointer.save(
                    thread_id=thread,
                    state=updated_state,
                    metadata={
                        "status": "verified",
                        "rule_ids": verification.details.get("rule_ids", []),
                        "task_desc": self.task_desc,
                    }
                )
                updated_state["ejentum_checkpoint_id"] = cp_id

        return updated_state


class EjentumConditionalEdge:
    """Routing function for LangGraph conditional edges based on Ejentum verification status."""

    def __init__(self, rollback_target: str = "replan", default_target: str = "execute"):
        self.rollback_target = rollback_target
        self.default_target = default_target

    def __call__(self, state: Dict[str, Any]) -> str:
        """Route to rollback target if verification is unsatisfied, else proceed to default."""
        if state.get("ejentum_is_satisfied") is False or state.get("ejentum_handshake_passed") is False:
            return self.rollback_target
        return state.get("next_step", self.default_target)


class EjentumHandshakeNode:
    """LangGraph node enforcing deterministic multi-agent handshake contracts between agent roles."""

    def __init__(
        self,
        contract: Union[HandshakeContract, str],
        client: Optional[Ejentum] = None,
        checkpointer: Optional[BaseCheckpointStore] = None,
        thread_id: Optional[str] = None,
        auto_rollback_state: bool = True,
        account_id: str = "anonymous",
    ):
        if client is not None:
            self.client = client
        elif os.getenv("EJENTUM_API_KEY"):
            self.client = Ejentum()
        else:
            self.client = Ejentum(local=True)
        if isinstance(contract, str):
            if "->" in contract:
                topo = get_preset_topology("planner_implementer_auditor")
                resolved = topo.contracts.get(contract)
                if not resolved:
                    raise ValueError(f"Unknown contract transition '{contract}' in default topology")
                self.contract = resolved
            else:
                topo = get_preset_topology(contract)
                self.contract = next(iter(topo.contracts.values()))
        else:
            self.contract = contract

        self.engine = HandshakeEngine(client=self.client)
        self.checkpointer = checkpointer
        self.thread_id = thread_id
        self.auto_rollback_state = auto_rollback_state
        self.account_id = account_id

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate multi-agent state transition against the contract."""
        source_output = (
            state.get(f"{self.contract.source_role}_output")
            or state.get("output")
            or state.get("last_tool_output")
            or state.get("result")
            or state.get("proposal")
        )
        context = state.get("task") or state.get("context") or ""
        rejection_key = f"ejentum_rejections_{self.contract.name}"
        rejection_count = int(state.get(rejection_key, 0))
        thread = self.thread_id or state.get("thread_id") or state.get("session_id") or "default_thread"

        result = self.engine.evaluate(
            contract=self.contract,
            source_output=source_output,
            context=str(context) if context else None,
            rejection_count=rejection_count,
            task_desc=str(context) if context else None,
            account_id=self.account_id,
        )

        updated_state = dict(state)
        updated_state["ejentum_handshake"] = result.to_dict()
        updated_state["ejentum_handshake_passed"] = result.passed

        if result.passed:
            updated_state[rejection_key] = 0
            updated_state["next_step"] = self.contract.target_role
            if self.checkpointer:
                cp_id = self.checkpointer.save(
                    thread_id=thread,
                    state=updated_state,
                    metadata={
                        "status": "handshake_accepted",
                        "contract": self.contract.name,
                        "source_role": self.contract.source_role,
                        "target_role": self.contract.target_role,
                        "attestation_id": result.attestation.get("receipt_id") if result.attestation else None,
                    }
                )
                updated_state["ejentum_checkpoint_id"] = cp_id
        else:
            updated_state[rejection_key] = result.rejection_count
            updated_state["ejentum_critique"] = result.critique_prompt
            if result.status == HandshakeStatus.ESCALATED:
                updated_state["next_step"] = self.contract.escalation_target
            else:
                if self.checkpointer and self.auto_rollback_state:
                    prior = self.checkpointer.rollback(thread_id=thread, steps=1)
                    if prior and isinstance(prior.get("state"), dict):
                        updated_state = dict(prior["state"])
                        updated_state["ejentum_handshake"] = result.to_dict()
                        updated_state["ejentum_handshake_passed"] = False
                        updated_state[rejection_key] = result.rejection_count
                        updated_state["ejentum_critique"] = result.critique_prompt
                        updated_state["ejentum_rolled_back_from"] = prior.get("checkpoint_id")
                updated_state["next_step"] = self.contract.source_role

        return updated_state


class EjentumVisualGuardNode:
    """LangGraph node enforcing visual and layout invariants for VLM, UI, and diagram agents."""

    def __init__(
        self,
        client: Optional[Ejentum] = None,
        checkpointer: Optional[BaseCheckpointStore] = None,
        thread_id: Optional[str] = None,
        auto_rollback_state: bool = True,
        enforce_spatial: bool = True,
        enforce_diagram: bool = True,
        enforce_sequence: bool = True,
        max_overlap_iou: float = 0.20,
        rollback_target: str = "replan",
        success_target: str = "continue",
    ):
        if client is not None:
            self.client = client
        elif os.getenv("EJENTUM_API_KEY"):
            self.client = Ejentum()
        else:
            self.client = Ejentum(local=True)

        self.checkpointer = checkpointer
        self.thread_id = thread_id
        self.auto_rollback_state = auto_rollback_state
        self.enforce_spatial = enforce_spatial
        self.enforce_diagram = enforce_diagram
        self.enforce_sequence = enforce_sequence
        self.max_overlap_iou = max_overlap_iou
        self.rollback_target = rollback_target
        self.success_target = success_target

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate visual, diagram, and temporal invariants across state."""
        violations: List[str] = []
        thread = self.thread_id or state.get("thread_id") or state.get("session_id") or "default_thread"

        # Check spatial layout if elements or bounding_boxes are present
        if self.enforce_spatial:
            elements = state.get("elements") or state.get("bounding_boxes") or state.get("layout")
            if isinstance(elements, list) and elements:
                constraints = state.get("directional_constraints")
                res = self.client.verify_spatial_layout(
                    elements=elements,
                    directional_constraints=constraints,
                    max_overlap_iou=self.max_overlap_iou,
                )
                if not res.is_valid:
                    violations.extend(res.violations)

        # Check diagram semantics if diagram is present
        if self.enforce_diagram:
            diagram = state.get("diagram") or state.get("mermaid") or state.get("flowchart")
            if diagram:
                expected = state.get("expected_nodes")
                d_res = self.client.verify_diagram_semantics(diagram, expected_nodes=expected, enforce_dag=True)
                if not d_res.is_valid:
                    violations.extend(d_res.violations)

        # Check frame sequence if frames are present
        if self.enforce_sequence:
            frames = state.get("frames") or state.get("trajectory")
            if isinstance(frames, list) and len(frames) >= 2:
                f_res = self.client.verify_frame_sequence(frames)
                if not f_res.is_valid:
                    violations.extend(f_res.violations)

        updated_state = dict(state)
        passed = (len(violations) == 0)
        updated_state["ejentum_visual_passed"] = passed
        updated_state["ejentum_visual_violations"] = violations

        if passed:
            updated_state["next_step"] = self.success_target
            if self.checkpointer:
                cp_id = self.checkpointer.save(
                    thread_id=thread,
                    state=updated_state,
                    metadata={"status": "visual_invariants_verified"}
                )
                updated_state["ejentum_checkpoint_id"] = cp_id
        else:
            if self.checkpointer and self.auto_rollback_state:
                prior = self.checkpointer.rollback(thread_id=thread, steps=1)
                if prior and isinstance(prior.get("state"), dict):
                    updated_state = dict(prior["state"])
                    updated_state["ejentum_visual_passed"] = False
                    updated_state["ejentum_visual_violations"] = violations
                    updated_state["ejentum_rolled_back_from"] = prior.get("checkpoint_id")
            updated_state["next_step"] = self.rollback_target

        return updated_state
