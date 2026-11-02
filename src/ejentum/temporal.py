"""
Ejentum Temporal Guard - Stateful Temporal Logic and Tool-Sequence Invariant Verification.
Enforces multi-step policy safety rules over autonomous agent trajectories.
Includes canonical action categories and synonym normalization.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Optional, Set


DEFAULT_ACTION_CATEGORIES: Dict[str, List[str]] = {
    "network_egress": [
        "http_export", "post_to_remote_endpoint", "send_webhook", "curl_post",
        "fetch_url", "socket_connect", "external_api_call", "export_data", "upload_file"
    ],
    "credential_access": [
        "read_credentials", "get_secret", "read_keys", "get_api_key",
        "dump_env", "read_token_file", "extract_auth", "fetch_credentials"
    ],
    "database_mutation": [
        "execute_transfer", "write_db", "delete_records", "drop_table",
        "truncate_table", "update_records", "insert_record", "transfer_funds"
    ],
    "system_command": [
        "execute_shell", "run_bash", "cmd_exec", "terminal_run", "spawn_process"
    ],
}


def action_matches(candidate: str, rule_action: str, category_map: Dict[str, Set[str]]) -> bool:
    c = candidate.strip().lower()
    r = rule_action.strip().lower()
    if c == r:
        return True

    # Check if rule_action is an entire category
    if r in category_map and c in category_map[r]:
        return True

    # Check if candidate and rule_action share any category
    for cat, members in category_map.items():
        c_in_cat = (c in members or c == cat)
        r_in_cat = (r in members or r == cat)
        if c_in_cat and r_in_cat:
            return True

    return False


@dataclass
class ActionRecord:
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    status: str = "success"
    step_index: int = 0


@dataclass
class TemporalVerificationResult:
    is_valid: bool
    violated_rule: Optional[str] = None
    explanation: str = "Trajectory satisfies all temporal invariants."
    violating_action: Optional[str] = None
    step_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "violated_rule": self.violated_rule,
            "explanation": self.explanation,
            "violating_action": self.violating_action,
            "step_index": self.step_index,
        }


class TemporalPolicyRule:
    """Base class for temporal logic policies over tool trajectories."""
    def check(
        self,
        candidate_action: str,
        candidate_params: Dict[str, Any],
        history: List[ActionRecord],
        category_map: Optional[Dict[str, Set[str]]] = None,
    ) -> Optional[str]:
        """Returns None if valid, or an error string if violated."""
        raise NotImplementedError


class RequirePriorRule(TemporalPolicyRule):
    """
    Specifies that candidate_action may ONLY execute if required_prior_action
    occurred in the execution history (optionally within the last max_steps).
    Supports category and alias expansion.
    """
    def __init__(self, target_action: str, required_prior_action: str, max_steps: Optional[int] = None):
        self.target_action = target_action.strip().lower()
        self.required_prior_action = required_prior_action.strip().lower()
        self.max_steps = max_steps

    def check(
        self,
        candidate_action: str,
        candidate_params: Dict[str, Any],
        history: List[ActionRecord],
        category_map: Optional[Dict[str, Set[str]]] = None,
    ) -> Optional[str]:
        cmap = category_map or {}
        if not action_matches(candidate_action, self.target_action, cmap):
            return None

        recent_history = history
        if self.max_steps is not None:
            recent_history = history[-self.max_steps:]

        found = any(
            action_matches(rec.action, self.required_prior_action, cmap) and rec.status == "success"
            for rec in recent_history
        )
        if not found:
            window_msg = f" within the last {self.max_steps} steps" if self.max_steps else ""
            return (
                f"Temporal violation: Action '{candidate_action}' requires prior successful action "
                f"'{self.required_prior_action}'{window_msg}, which was not found in trajectory."
            )
        return None


class ForbidFollowingRule(TemporalPolicyRule):
    """
    Specifies that forbidden_action may NOT execute if prior_action occurred
    in execution history (optionally within max_steps).
    Prevents privilege escalation and unsafe multi-step exfiltration across tool aliases.
    """
    def __init__(self, prior_action: str, forbidden_action: str, max_steps: Optional[int] = None):
        self.prior_action = prior_action.strip().lower()
        self.forbidden_action = forbidden_action.strip().lower()
        self.max_steps = max_steps

    def check(
        self,
        candidate_action: str,
        candidate_params: Dict[str, Any],
        history: List[ActionRecord],
        category_map: Optional[Dict[str, Set[str]]] = None,
    ) -> Optional[str]:
        cmap = category_map or {}
        if not action_matches(candidate_action, self.forbidden_action, cmap):
            return None

        recent_history = history
        if self.max_steps is not None:
            recent_history = history[-self.max_steps:]

        prior_match = any(
            action_matches(rec.action, self.prior_action, cmap)
            for rec in recent_history
        )
        if prior_match:
            window_msg = f" within {self.max_steps} steps" if self.max_steps else ""
            return (
                f"Temporal violation: Action '{candidate_action}' (category-matched to '{self.forbidden_action}') "
                f"is forbidden following action '{self.prior_action}'{window_msg}."
            )
        return None


class RequireParamConditionRule(TemporalPolicyRule):
    """
    Specifies that parameters of candidate_action must satisfy a predicate condition.
    """
    def __init__(self, target_action: str, condition_fn: Callable[[Dict[str, Any]], bool], description: str):
        self.target_action = target_action.strip().lower()
        self.condition_fn = condition_fn
        self.description = description

    def check(
        self,
        candidate_action: str,
        candidate_params: Dict[str, Any],
        history: List[ActionRecord],
        category_map: Optional[Dict[str, Set[str]]] = None,
    ) -> Optional[str]:
        cmap = category_map or {}
        if not action_matches(candidate_action, self.target_action, cmap):
            return None

        try:
            passed = bool(self.condition_fn(candidate_params))
        except Exception as e:
            return f"Parameter validation error for '{candidate_action}': {str(e)}"

        if not passed:
            return (
                f"Temporal parameter invariant violated for '{candidate_action}': "
                f"{self.description}"
            )
        return None


class RequireApprovalRule(TemporalPolicyRule):
    """
    Specifies that high-risk actions require explicit human/admin confirmation or signed token.
    """
    def __init__(self, target_action: str, approval_key: str = "approved"):
        self.target_action = target_action.strip().lower()
        self.approval_key = approval_key

    def check(
        self,
        candidate_action: str,
        candidate_params: Dict[str, Any],
        history: List[ActionRecord],
        category_map: Optional[Dict[str, Set[str]]] = None,
    ) -> Optional[str]:
        cmap = category_map or {}
        if not action_matches(candidate_action, self.target_action, cmap):
            return None

        approved = candidate_params.get(self.approval_key)
        if approved is not True:
            return (
                f"Temporal authorization violation: Action '{candidate_action}' requires explicit "
                f"approval token parameter '{self.approval_key}: true'."
            )
        return None


class TemporalGuard:
    """
    Stateful runtime verification engine tracking multi-step tool calls
    and verifying temporal safety logic before actions execute.
    Includes canonical action categories and alias normalization.
    """
    def __init__(
        self,
        rules: Optional[List[TemporalPolicyRule]] = None,
        categories: Optional[Dict[str, List[str]]] = None,
    ):
        self.rules: List[TemporalPolicyRule] = rules or []
        self.history: List[ActionRecord] = []
        self.category_map: Dict[str, Set[str]] = {
            k: {item.strip().lower() for item in v}
            for k, v in DEFAULT_ACTION_CATEGORIES.items()
        }
        if categories:
            for cat, actions in categories.items():
                self.register_category(cat, actions)

    def register_category(self, category_name: str, actions: List[str]) -> "TemporalGuard":
        cat_key = category_name.strip().lower()
        self.category_map.setdefault(cat_key, set()).update(a.strip().lower() for a in actions)
        return self

    def register_alias(self, canonical_action: str, alias: str) -> "TemporalGuard":
        canon = canonical_action.strip().lower()
        al = alias.strip().lower()
        for cat, members in self.category_map.items():
            if canon in members or canon == cat:
                members.add(al)
        return self

    def add_rule(self, rule: TemporalPolicyRule) -> "TemporalGuard":
        self.rules.append(rule)
        return self

    def require_prior(self, target_action: str, required_prior_action: str, max_steps: Optional[int] = None) -> "TemporalGuard":
        return self.add_rule(RequirePriorRule(target_action, required_prior_action, max_steps))

    def forbid_following(self, prior_action: str, forbidden_action: str, max_steps: Optional[int] = None) -> "TemporalGuard":
        return self.add_rule(ForbidFollowingRule(prior_action, forbidden_action, max_steps))

    def require_param_condition(self, target_action: str, condition_fn: Callable[[Dict[str, Any]], bool], description: str) -> "TemporalGuard":
        return self.add_rule(RequireParamConditionRule(target_action, condition_fn, description))

    def require_approval(self, target_action: str, approval_key: str = "approved") -> "TemporalGuard":
        return self.add_rule(RequireApprovalRule(target_action, approval_key))

    def record_action(self, action: str, parameters: Optional[Dict[str, Any]] = None, status: str = "success") -> ActionRecord:
        rec = ActionRecord(
            action=action,
            parameters=parameters or {},
            status=status,
            step_index=len(self.history),
        )
        self.history.append(rec)
        return rec

    def verify_action(self, candidate_action: str, parameters: Optional[Dict[str, Any]] = None) -> TemporalVerificationResult:
        params = parameters or {}
        for rule in self.rules:
            error_msg = rule.check(candidate_action, params, self.history, self.category_map)
            if error_msg:
                return TemporalVerificationResult(
                    is_valid=False,
                    violated_rule=rule.__class__.__name__,
                    explanation=error_msg,
                    violating_action=candidate_action,
                    step_index=len(self.history),
                )
        return TemporalVerificationResult(
            is_valid=True,
            explanation=f"Action '{candidate_action}' conforms to all temporal invariants.",
            violating_action=None,
            step_index=len(self.history),
        )

    def verify_trajectory(self, trajectory: List[Dict[str, Any]]) -> TemporalVerificationResult:
        """
        Verifies a complete sequence of actions offline or during audit.
        Each item in trajectory must have 'action' and optional 'parameters' and 'status'.
        """
        temp_history: List[ActionRecord] = []
        for idx, item in enumerate(trajectory):
            action_name = item.get("action", "")
            params = item.get("parameters", {})
            status = item.get("status", "success")

            for rule in self.rules:
                err = rule.check(action_name, params, temp_history, self.category_map)
                if err:
                    return TemporalVerificationResult(
                        is_valid=False,
                        violated_rule=rule.__class__.__name__,
                        explanation=err,
                        violating_action=action_name,
                        step_index=idx,
                    )

            temp_history.append(ActionRecord(
                action=action_name,
                parameters=params,
                status=status,
                step_index=idx,
            ))

        return TemporalVerificationResult(
            is_valid=True,
            explanation="Complete trajectory conforms to all temporal invariants.",
            step_index=len(trajectory),
        )

    def reset(self) -> None:
        self.history.clear()
