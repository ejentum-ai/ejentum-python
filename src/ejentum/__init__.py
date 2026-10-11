"""
Ejentum Python SDK — Deterministic Reasoning-Augmented Retrieval (RAR)
"""

from .client import Ejentum
from .exceptions import APIError, AuthenticationError, EjentumError, ScaffoldNotFoundError
from .wrappers import wrap_anthropic, wrap_openai

__version__ = "0.1.0"
__all__ = [
    "Ejentum",
    "wrap_openai",
    "wrap_anthropic",
    "EjentumError",
    "AuthenticationError",
    "APIError",
    "ScaffoldNotFoundError",
]
