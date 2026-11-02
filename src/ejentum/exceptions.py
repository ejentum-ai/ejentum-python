class EjentumError(Exception):
    """Base exception for all Ejentum SDK errors."""
    pass


class AuthenticationError(EjentumError):
    """Raised when an API key is missing or rejected (HTTP 401)."""
    pass


class APIError(EjentumError):
    """Raised when the Ejentum API returns an unexpected error."""
    def __init__(self, message: str, status_code: int = 500, response_body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class ScaffoldNotFoundError(EjentumError):
    """Raised when the Ejentum engine cannot produce a valid scaffold."""
    pass


class EjentumSecurityError(EjentumError):
    """Raised when fail_closed policy blocks execution due to API unavailability or policy tripwire."""
    pass


class EvidenceVerificationResult:
    """Result of validating execution output against mandatory evidence contracts."""
    def __init__(
        self,
        is_satisfied: bool,
        status: str,
        missing_evidence: list,
        observed_evidence: dict,
        violations: list,
        details: dict = None,
    ):
        self.is_satisfied = is_satisfied
        self.status = status
        self.missing_evidence = missing_evidence
        self.observed_evidence = observed_evidence
        self.violations = violations
        self.details = details or {}

    def to_dict(self) -> dict:
        return {
            "is_satisfied": self.is_satisfied,
            "status": self.status,
            "missing_evidence": self.missing_evidence,
            "observed_evidence": self.observed_evidence,
            "violations": self.violations,
            "details": self.details,
        }

    def __repr__(self) -> str:
        return (
            f"<EvidenceVerificationResult status={self.status} "
            f"is_satisfied={self.is_satisfied} missing={len(self.missing_evidence)} "
            f"violations={len(self.violations)}>"
        )


