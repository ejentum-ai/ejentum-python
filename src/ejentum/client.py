"""
Official Python client for the Ejentum Deterministic Reasoning-Augmented Retrieval (RAR) API.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from .exceptions import APIError, AuthenticationError, ScaffoldNotFoundError

DEFAULT_BASE_URL = "https://api.ejentum.com/logicv1/"
SDK_VERSION = "1.4.0"


class Ejentum:
    """Ejentum Client for deterministic cognitive scaffold retrieval and prompt augmentation.

    Example:
        >>> from ejentum import Ejentum
        >>> client = Ejentum()  # Reads EJENTUM_API_KEY from environment
        >>> scaffold = client.scaffold("diagnose database connection pool exhaustion")
        >>> augmented_prompt = client.augment("Why is checkout failing?")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 10.0,
    ) -> None:
        self.api_key = api_key or os.getenv("EJENTUM_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                "Ejentum API key is missing. Pass `api_key='...'` or set the "
                "`EJENTUM_API_KEY` environment variable. Get a key at https://ejentum.com/pricing"
            )
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout

    def get_ability(self, query: str, mode: str = "reasoning") -> Dict[str, Any]:
        """Fetch the full raw cognitive ability payload from the Ejentum engine."""
        payload = json.dumps({"query": query, "mode": mode}).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"Ejentum-Python-SDK/{SDK_VERSION}",
        }

        req = urllib.request.Request(self.base_url, data=payload, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data
        except urllib.error.HTTPError as e:
            raw_body = e.read().decode("utf-8", errors="replace")
            if e.code == 401:
                raise AuthenticationError(
                    f"Ejentum authentication failed (401 Unauthorized): {raw_body}"
                ) from e
            raise APIError(
                f"Ejentum API request failed with status {e.code}: {raw_body}",
                status_code=e.code,
                response_body=raw_body,
            ) from e
        except urllib.error.URLError as e:
            raise APIError(f"Network error connecting to Ejentum API: {e.reason}") from e
        except Exception as e:
            raise APIError(f"Unexpected error communicating with Ejentum: {str(e)}") from e

    def scaffold(self, query: str, mode: str = "reasoning") -> str:
        """Fetch the extracted cognitive scaffold string for a given task query.

        Args:
            query: The user task, error message, or diagnostic query.
            mode: Reasoning mode (e.g., 'reasoning', 'causal', 'diagnostic', 'graph').

        Returns:
            The raw cognitive scaffold text ready for system prompt injection.
        """
        data = self.get_ability(query=query, mode=mode)

        if isinstance(data, dict):
            if "graph_scaffold" in data and isinstance(data["graph_scaffold"], str):
                return data["graph_scaffold"].strip()
            if "single_ability" in data and isinstance(data["single_ability"], str):
                return data["single_ability"].strip()
            if "scaffold" in data and isinstance(data["scaffold"], str):
                return data["scaffold"].strip()

        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            val = data[0].get(mode) or data[0].get("reasoning") or data[0].get("single_ability")
            if isinstance(val, str) and val.strip():
                return val.strip()

        raise ScaffoldNotFoundError(
            f"Ejentum API did not return a valid '{mode}' scaffold for query: {query!r}"
        )

    def augment(self, prompt: str, mode: str = "reasoning") -> str:
        """Prepend a cognitive scaffold block to a raw prompt string."""
        scaffold_text = self.scaffold(query=prompt, mode=mode)
        instruction = (
            "Apply the cognitive scaffold below, then answer the user's task.\n\n"
            f"[COGNITIVE SCAFFOLD]\n{scaffold_text}\n[END SCAFFOLD]\n\n"
        )
        return f"{instruction}{prompt}"

    def augment_messages(
        self, messages: List[Dict[str, Any]], mode: str = "reasoning"
    ) -> List[Dict[str, Any]]:
        """Inject a cognitive scaffold into an OpenAI/Anthropic messages array.

        Finds the last user message to use as the query, then prepends the scaffold
        instruction to the system message (or creates one at index 0).
        """
        if not messages:
            return messages

        # Find last user prompt as query
        user_query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_query = content
                elif isinstance(content, list):
                    # Multi-modal / block format
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            user_query = part.get("text", "")
                            break
                break

        if not user_query:
            user_query = "analyze and execute instructions"

        scaffold_text = self.scaffold(query=user_query, mode=mode)
        instruction = (
            "Apply the cognitive scaffold below, then execute the user's task.\n\n"
            f"[COGNITIVE SCAFFOLD]\n{scaffold_text}\n[END SCAFFOLD]"
        )

        new_messages = [dict(m) for m in messages]

        if new_messages[0].get("role") == "system":
            existing_sys = new_messages[0].get("content", "")
            new_messages[0]["content"] = f"{instruction}\n\n{existing_sys}"
        else:
            new_messages.insert(0, {"role": "system", "content": instruction})

        return new_messages

    def graph(self, query: str, depth: int = 1) -> Dict[str, Any]:
        """Retrieve a multi-hop Cognitive Directed Acyclic Graph (DAG) across the 933-edge synergy graph.

        Args:
            query: The complex task, system failure, or architecture question.
            depth: Traversal depth (1 or 2, default: 1).
                   - depth=1: Immediate requires, enhances, and fallback nodes.
                   - depth=2: Recursive multi-hop DAG with 4-stage execution planning:
                     prerequisite_invariants -> root_operator -> cognitive_amplification -> resiliency_contingency.

        Returns:
            Dict containing root, requires, enhances, fallback abilities, edges, execution_plan (depth=2), and formatted scaffold.
        """
        depth = max(1, min(int(depth), 2))
        payload = json.dumps({"query": query, "mode": "graph", "depth": depth}).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"Ejentum-Python-SDK/{SDK_VERSION}",
        }
        req = urllib.request.Request(self.base_url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw_body = e.read().decode("utf-8", errors="replace")
            if e.code == 401:
                raise AuthenticationError(
                    f"Ejentum authentication failed (401 Unauthorized): {raw_body}"
                ) from e
            raise APIError(
                f"Ejentum graph request failed with status {e.code}: {raw_body}",
                status_code=e.code,
                response_body=raw_body,
            ) from e
        except urllib.error.URLError as e:
            raise APIError(f"Network error connecting to Ejentum API: {e.reason}") from e
        except Exception as e:
            raise APIError(f"Ejentum graph request failed: {e}") from e

    def fts_search(self, query: str, limit: int = 4) -> Dict[str, Any]:
        """Perform sub-millisecond SQLite FTS5 full-text keyword search across 311 reasoning abilities.

        Executes high-speed BM25 full-text ranking directly against the pre-compiled SQLite FTS5
        database on the server.

        Args:
            query: Keyword query or search terms (e.g., 'deadlock circular dependency sequence').
            limit: Maximum number of abilities to return (default: 4).

        Returns:
            Dict containing status, query, count, engine ('sqlite_fts5'), and ranked ability results:
            {
                "status": "success",
                "engine": "sqlite_fts5",
                "query": "...",
                "count": 3,
                "results": [
                    {
                        "ability_id": "TE-004",
                        "ability_name": "Deadlock Breaker",
                        "domain": "Temporal",
                        "graph_op": "BREAK_DEADLOCK",
                        "prompt_override": "...",
                        "rank": -18.966
                    }, ...
                ]
            }
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("Query cannot be empty for FTS search")

        base = self.base_url.rstrip("/")
        if base.endswith("/logicv1"):
            base_endpoint = f"{base}/search/fts"
        else:
            base_endpoint = f"{base}/logicv1/search/fts"

        encoded_query = urllib.parse.quote(query)
        url = f"{base_endpoint}?query={encoded_query}&limit={int(limit)}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": f"Ejentum-Python-SDK/{SDK_VERSION}",
        }
        req = urllib.request.Request(url, headers=headers, method="GET")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw_body = e.read().decode("utf-8", errors="replace")
            if e.code == 401:
                raise AuthenticationError(
                    f"Ejentum authentication failed (401 Unauthorized): {raw_body}"
                ) from e
            raise APIError(
                f"Ejentum FTS search failed with status {e.code}: {raw_body}",
                status_code=e.code,
                response_body=raw_body,
            ) from e
        except urllib.error.URLError as e:
            raise APIError(f"Network error connecting to Ejentum API: {e.reason}") from e
        except Exception as e:
            raise APIError(f"Unexpected error communicating with Ejentum: {str(e)}") from e

    def feedback(
        self,
        ability_id: str,
        outcome: str = "success",
        score: float = 1.0,
        query: str = "",
        comment: str = ""
    ) -> Dict[str, Any]:
        """Submit execution telemetry to calibrate adaptive retrieval weights.

        Args:
            ability_id: The ID of the ability that was executed (e.g., 'CA-025').
            outcome: 'success' or 'failure'.
            score: Performance score between 0.0 and 1.0.
            query: The original query text.
            comment: Optional qualitative evaluation note.

        Returns:
            Dict confirming feedback recording and calibrated retrieval weight.
        """
        feedback_url = self.base_url.rstrip("/")
        if feedback_url.endswith("/logicv1"):
            feedback_url += "/feedback"
        else:
            feedback_url += "/logicv1/feedback"

        payload = json.dumps({
            "ability_id": ability_id,
            "outcome": outcome,
            "score": score,
            "query": query,
            "comment": comment
        }).encode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"Ejentum-Python-SDK/{SDK_VERSION}",
        }
        req = urllib.request.Request(feedback_url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw_body = e.read().decode("utf-8", errors="replace")
            if e.code == 401:
                raise AuthenticationError(
                    f"Ejentum authentication failed (401 Unauthorized): {raw_body}"
                ) from e
            raise APIError(
                f"Ejentum feedback submission failed with status {e.code}: {raw_body}",
                status_code=e.code,
                response_body=raw_body,
            ) from e
        except urllib.error.URLError as e:
            raise APIError(f"Network error connecting to Ejentum API: {e.reason}") from e
        except Exception as e:
            raise APIError(f"Ejentum feedback submission failed: {e}") from e
