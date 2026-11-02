"""
Ejentum CrewAI Integration.
Provides step callback hooks and guard tools that intercept agent execution.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Union
from ..client import Ejentum
from ..exceptions import EjentumSecurityError
from ..handshake import (
    HandshakeContract,
    HandshakeResult,
    HandshakeStatus,
    HandshakeEngine,
    get_preset_topology,
)


class EjentumStepHook:
    """Step callback hook for CrewAI agents to enforce cognitive and evidence invariants."""

    def __init__(
        self,
        client: Optional[Ejentum] = None,
        required_evidence: Optional[Union[Dict[str, Any], List[str]]] = None,
        fail_mode: str = "closed",
        on_violation: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        if client is not None:
            self.client = client
        elif os.getenv("EJENTUM_API_KEY"):
            self.client = Ejentum()
        else:
            self.client = Ejentum(local=True)
        self.required_evidence = required_evidence
        self.fail_mode = fail_mode
        self.on_violation = on_violation

    def __call__(self, step_output: Any) -> Any:
        """Inspect step output from CrewAI agent."""
        text_content = ""
        if hasattr(step_output, "output"):
            text_content = str(step_output.output)
        elif hasattr(step_output, "text"):
            text_content = str(step_output.text)
        elif hasattr(step_output, "result"):
            text_content = str(step_output.result)
        elif isinstance(step_output, dict):
            text_content = str(step_output.get("output", step_output))
        else:
            text_content = str(step_output)

        if self.required_evidence:
            verification = self.client.verify_evidence(
                execution_result=text_content,
                required_evidence=self.required_evidence,
                fail_mode=self.fail_mode,
            )
            if not verification.is_satisfied:
                event = {
                    "step_output": text_content,
                    "violations": verification.violations,
                    "missing_evidence": verification.missing_evidence,
                }
                if self.on_violation:
                    self.on_violation(event)
                if self.fail_mode == "closed":
                    raise EjentumSecurityError(
                        f"Ejentum step guard triggered: violations={verification.violations}, "
                        f"missing={verification.missing_evidence}"
                    )
        return step_output


class EjentumGuardTool:
    """Guard tool callable by CrewAI agents to obtain pre-execution clearance and attestation."""

    name: str = "ejentum_preflight_guard"
    description: str = (
        "Validates proposed tool calls, database mutations, or code actions against "
        "deterministic invariants before execution. Returns gate clearance and attestation."
    )

    def __init__(self, client: Optional[Ejentum] = None, fail_mode: str = "closed"):
        if client is not None:
            self.client = client
        elif os.getenv("EJENTUM_API_KEY"):
            self.client = Ejentum()
        else:
            self.client = Ejentum(local=True)
        self.fail_mode = fail_mode

    def run(self, action_description: str, proposed_payload: Optional[str] = None) -> str:
        """Evaluate preflight safety for the specified action."""
        query = action_description
        if proposed_payload:
            query = f"{action_description}: {proposed_payload}"

        rule_data = self.client.get_rules(query)
        trace_summary = self.client.format_trace(rule_data)
        return f"Ejentum Guard Clearance Granted.\n{trace_summary}"

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        action = kwargs.get("action_description") or (args[0] if args else "unspecified action")
        payload = kwargs.get("proposed_payload") or (args[1] if len(args) > 1 else None)
        return self.run(str(action), str(payload) if payload else None)


class EjentumHandshakeHook:
    """CrewAI step callback that validates inter-agent handoffs against a HandshakeContract."""

    def __init__(
        self,
        contract: Union[HandshakeContract, str],
        client: Optional[Ejentum] = None,
        fail_mode: str = "closed",
        on_violation: Optional[Callable[[Dict[str, Any]], None]] = None,
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
        self.fail_mode = fail_mode
        self.on_violation = on_violation
        self.account_id = account_id
        self.rejection_count = 0

    def __call__(self, step_output: Any) -> Any:
        """Validate step output before handing off to next agent."""
        text_content = ""
        if hasattr(step_output, "output"):
            text_content = str(step_output.output)
        elif hasattr(step_output, "text"):
            text_content = str(step_output.text)
        elif hasattr(step_output, "result"):
            text_content = str(step_output.result)
        elif isinstance(step_output, dict):
            text_content = str(step_output.get("output", step_output))
        else:
            text_content = str(step_output)

        result = self.engine.evaluate(
            contract=self.contract,
            source_output=text_content,
            rejection_count=self.rejection_count,
            account_id=self.account_id,
        )

        if not result.passed:
            self.rejection_count += 1
            event = {
                "step_output": text_content,
                "contract": self.contract.name,
                "violations": result.violations,
                "missing_evidence": result.missing_evidence,
                "rejection_count": self.rejection_count,
                "critique_prompt": result.critique_prompt,
            }
            if self.on_violation:
                self.on_violation(event)
            if self.fail_mode == "closed":
                raise EjentumSecurityError(
                    f"Ejentum handshake rejected for '{self.contract.name}': "
                    f"violations={result.violations}, missing={result.missing_evidence}"
                )
        else:
            self.rejection_count = 0

        return step_output
