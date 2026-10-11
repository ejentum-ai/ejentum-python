"""
Drop-in wrappers for popular LLM provider clients (OpenAI, Anthropic).
Automatically intercepts chat completion calls to inject Ejentum cognitive scaffolds in <80ms.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

from .client import Ejentum


def wrap_openai(client: Any, ejentum_client: Optional[Ejentum] = None) -> Any:
    """Wrap an official OpenAI client to automatically scaffold chat completions.

    Example:
        >>> from openai import OpenAI
        >>> from ejentum import wrap_openai
        >>> client = wrap_openai(OpenAI())
        >>> response = client.chat.completions.create(
        ...     model="gpt-4o",
        ...     messages=[{"role": "user", "content": "Diagnose latency spike"}],
        ...     ejentum_mode="reasoning",
        ... )
    """
    if ejentum_client is None:
        ejentum_client = Ejentum()

    orig_create = client.chat.completions.create

    @functools.wraps(orig_create)
    def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        mode = kwargs.pop("ejentum_mode", "reasoning")
        enabled = kwargs.pop("ejentum_enabled", True)

        if enabled and "messages" in kwargs:
            kwargs["messages"] = ejentum_client.augment_messages(
                messages=kwargs["messages"], mode=mode
            )

        return orig_create(*args, **kwargs)

    client.chat.completions.create = wrapped_create
    return client


def wrap_anthropic(client: Any, ejentum_client: Optional[Ejentum] = None) -> Any:
    """Wrap an official Anthropic client to automatically scaffold message creations.

    Example:
        >>> from anthropic import Anthropic
        >>> from ejentum import wrap_anthropic
        >>> client = wrap_anthropic(Anthropic())
        >>> response = client.messages.create(
        ...     model="claude-3-7-sonnet-20250219",
        ...     max_tokens=1024,
        ...     messages=[{"role": "user", "content": "Diagnose latency spike"}],
        ...     ejentum_mode="reasoning",
        ... )
    """
    if ejentum_client is None:
        ejentum_client = Ejentum()

    orig_create = client.messages.create

    @functools.wraps(orig_create)
    def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        mode = kwargs.pop("ejentum_mode", "reasoning")
        enabled = kwargs.pop("ejentum_enabled", True)

        if enabled and "messages" in kwargs:
            # Get last user prompt for scaffold query
            user_query = ""
            for msg in reversed(kwargs["messages"]):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        user_query = content
                    break

            if not user_query:
                user_query = "execute instructions"

            scaffold_text = ejentum_client.scaffold(query=user_query, mode=mode)
            instruction = (
                "Apply the cognitive scaffold below, then execute the user's task.\n\n"
                f"[COGNITIVE SCAFFOLD]\n{scaffold_text}\n[END SCAFFOLD]"
            )

            # Anthropic takes system as a top-level parameter
            existing_system = kwargs.get("system", "")
            if existing_system:
                kwargs["system"] = f"{instruction}\n\n{existing_system}"
            else:
                kwargs["system"] = instruction

        return orig_create(*args, **kwargs)

    client.messages.create = wrapped_create
    return client
