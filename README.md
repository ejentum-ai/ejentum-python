# Ejentum Python SDK

[![PyPI Version](https://img.shields.io/badge/pypi-1.4.0-blue.svg)](https://pypi.org/project/ejentum/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Latency](https://img.shields.io/badge/API%20Latency-%3C80ms-neon)](https://api.ejentum.com)

The official Python SDK for **Ejentum** — Deterministic Reasoning-Augmented Retrieval (RAR) Cognitive Infrastructure.

Ejentum indexes 300+ deterministic mental models and cognitive scaffolds. In `<80ms`, it retrieves and injects the optimal reasoning scaffold into your LLM prompt, eliminating hallucinated leaps, sycophantic shortcuts, and premature convergence in production AI agents.

---

## ⚡ Installation

```bash
pip install ejentum
```

*Zero dependencies required. Uses Python standard library only.*

---

## 🔑 Authentication

Export your API key in your environment (or pass `api_key="..."` directly to the client):

```bash
export EJENTUM_API_KEY="ej_live_your_key_here"
```

*Need an API key? Get instant access at [ejentum.com/pricing](https://ejentum.com/pricing).*

---

## 🚀 Quickstart

### 1. Basic Scaffold Retrieval

```python
from ejentum import Ejentum

client = Ejentum()

# Retrieve the deterministic cognitive scaffold for any query (<80ms)
scaffold = client.scaffold("diagnose PostgreSQL high CPU and lock contention")
print(scaffold)
```

### 2. Auto-Augment a Prompt

```python
# Automatically prepends the scaffold instruction block to your task prompt
prompt = "Design a failover strategy for our payment processing service."
augmented_prompt = client.augment(prompt, mode="reasoning")

print(augmented_prompt)
# Output:
# Apply the cognitive scaffold below, then answer the user's task.
# [COGNITIVE SCAFFOLD]
# ... [Deterministic Causal / Resiliency Scaffold] ...
# [END SCAFFOLD]
# Design a failover strategy for our payment processing service.
```

### 3. Recursive Multi-Hop Cognitive DAG (`depth=2`)

For complex agent workflows, generate a recursive Directed Acyclic Graph (DAG) across the 933-edge synergy graph with an automated 4-stage execution plan:

```python
# Retrieve 2-hop topological execution plan & compound scaffold
graph_plan = client.graph("Migrate 50M rows across sharded PostgreSQL database", depth=2)

print("Root Operator:", graph_plan["dag"]["root"])
print("Execution Stages:", len(graph_plan["dag"]["execution_plan"]))
for stage in graph_plan["dag"]["execution_plan"]:
    print(f"  Stage {stage['stage']} ({stage['phase']}): {stage['abilities']}")

# Injects 4-stage cognitive scaffold (prerequisites -> root -> compounding multipliers -> fallbacks)
scaffold = graph_plan["graph_scaffold"]
```

### 4. Sub-Millisecond SQLite FTS5 Full-Text Search

Perform lightning-fast full-text keyword searches with BM25 ranking across the 311 ability catalog:

```python
# Sub-1ms SQLite FTS5 query
results = client.fts_search("deadlock circular dependency sequence", limit=3)

for match in results["results"]:
    print(f"[{match['ability_id']}] {match['ability_name']} ({match['domain']}) - Rank: {match['rank']}")
```

---

## 🔌 1-Line Drop-in Wrappers

### OpenAI Drop-in

Wrap your existing `OpenAI()` client. All `chat.completions.create` calls will automatically fetch and inject the optimal Ejentum scaffold before invoking OpenAI:

```python
from openai import OpenAI
from ejentum import wrap_openai

# Wrap your existing client in 1 line:
client = wrap_openai(OpenAI())

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Diagnose silent failure in checkout webhooks"}
    ],
    ejentum_mode="reasoning"  # Optional: defaults to "reasoning"
)

print(response.choices[0].message.content)
```

### Anthropic Drop-in

```python
from anthropic import Anthropic
from ejentum import wrap_anthropic

client = wrap_anthropic(Anthropic())

response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Refactor auth token refresh to prevent race condition"}
    ]
)

print(response.content[0].text)
```

### LangChain / Agent Loop Integration

```python
from ejentum import Ejentum
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

ejentum = Ejentum()

def scaffold_runnable(inputs):
    scaffold = ejentum.scaffold(inputs["task"])
    return {
        "system_scaffold": f"Apply this scaffold:\n{scaffold}",
        "task": inputs["task"]
    }

prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_scaffold}"),
    ("user", "{task}")
])

chain = scaffold_runnable | prompt | ChatOpenAI(model="gpt-4o")
result = chain.invoke({"task": "Plan a zero-downtime database migration for 50M rows."})
```

---

## 🛠️ API Reference

### `Ejentum(api_key=None, base_url="https://api.ejentum.com/logicv1/", timeout=10.0)`
* `scaffold(query: str, mode="reasoning") -> str`: Returns the raw cognitive scaffold string for system prompt injection.
* `graph(query: str, depth: int = 1) -> dict`: Returns the Cognitive DAG across the 933-edge synergy graph. Supports `depth=1` and `depth=2` (4-stage recursive execution plan).
* `fts_search(query: str, limit: int = 4) -> dict`: Executes sub-millisecond SQLite FTS5 full-text keyword search and BM25 ranking.
* `augment(prompt: str, mode="reasoning") -> str`: Prepends the cognitive scaffold instruction block directly to your prompt string.
* `augment_messages(messages: list[dict], mode="reasoning") -> list[dict]`: Augments an OpenAI/Anthropic messages list in-place or prepends to the system message.
* `feedback(ability_id: str, outcome="success", score=1.0, query="", comment="") -> dict`: Submits execution telemetry to calibrate adaptive retrieval weights.
* `get_ability(query: str, mode="reasoning") -> dict`: Returns the full metadata payload including `ability_id` and matched `domain`.

---

## 📄 License
MIT © [Ejentum.com](https://ejentum.com)
