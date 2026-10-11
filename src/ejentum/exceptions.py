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
