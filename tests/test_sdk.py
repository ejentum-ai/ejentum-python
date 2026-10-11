import json
import os
import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

# Ensure sdk/python/src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ejentum.client import Ejentum
from ejentum.exceptions import APIError, AuthenticationError
from ejentum.wrappers import wrap_anthropic, wrap_openai


class TestEjentumSDK(unittest.TestCase):
    def setUp(self):
        self.api_key = "ej_live_test_dummy_key_123"

    def test_missing_api_key_raises_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(AuthenticationError):
                Ejentum()

    def test_init_with_param_or_env(self):
        client = Ejentum(api_key=self.api_key)
        self.assertEqual(client.api_key, self.api_key)

        with patch.dict(os.environ, {"EJENTUM_API_KEY": "ej_live_env_key"}):
            client_env = Ejentum()
            self.assertEqual(client_env.api_key, "ej_live_env_key")

    @patch("urllib.request.urlopen")
    def test_scaffold_retrieval(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "status": "success",
            "domain": "Causal",
            "ability_id": "CA-001",
            "single_ability": "Scaffold: Isolate root causes using five whys."
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        client = Ejentum(api_key=self.api_key)
        scaffold = client.scaffold("diagnose error")

        self.assertEqual(scaffold, "Scaffold: Isolate root causes using five whys.")

    @patch("urllib.request.urlopen")
    def test_augment_prompt(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "single_ability": "Analyze dependencies carefully."
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        client = Ejentum(api_key=self.api_key)
        augmented = client.augment("Migrate database")

        self.assertIn("[COGNITIVE SCAFFOLD]", augmented)
        self.assertIn("Analyze dependencies carefully.", augmented)
        self.assertIn("[END SCAFFOLD]", augmented)
        self.assertTrue(augmented.endswith("Migrate database"))

    @patch("urllib.request.urlopen")
    def test_augment_messages_without_system_prompt(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "single_ability": "Check latency."
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        client = Ejentum(api_key=self.api_key)
        original_messages = [{"role": "user", "content": "Fix timeout"}]
        augmented = client.augment_messages(original_messages)

        self.assertEqual(len(augmented), 2)
        self.assertEqual(augmented[0]["role"], "system")
        self.assertIn("Check latency.", augmented[0]["content"])
        self.assertEqual(augmented[1]["role"], "user")
        self.assertEqual(augmented[1]["content"], "Fix timeout")

    @patch("urllib.request.urlopen")
    def test_augment_messages_with_existing_system_prompt(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "single_ability": "Check latency."
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        client = Ejentum(api_key=self.api_key)
        original_messages = [
            {"role": "system", "content": "You are a senior site reliability engineer."},
            {"role": "user", "content": "Fix timeout"}
        ]
        augmented = client.augment_messages(original_messages)

        self.assertEqual(len(augmented), 2)
        self.assertEqual(augmented[0]["role"], "system")
        self.assertIn("Check latency.", augmented[0]["content"])
        self.assertIn("You are a senior site reliability engineer.", augmented[0]["content"])
        self.assertEqual(augmented[1]["role"], "user")

    @patch("urllib.request.urlopen")
    def test_wrap_openai(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "single_ability": "Scaffold: Test bounds."
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        mock_orig_create = MagicMock(return_value={"choices": [{"message": {"content": "ok"}}]})
        class MockCompletions:
            def __init__(self):
                self.create = mock_orig_create

        class MockChat:
            def __init__(self):
                self.completions = MockCompletions()

        class MockOpenAI:
            def __init__(self):
                self.chat = MockChat()

        fake_openai = MockOpenAI()
        ejentum = Ejentum(api_key=self.api_key)
        wrapped = wrap_openai(fake_openai, ejentum_client=ejentum)

        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Write unit tests"}]
        )

        mock_orig_create.assert_called_once()
        passed_kwargs = mock_orig_create.call_args[1]
        self.assertEqual(passed_kwargs["model"], "gpt-4o")
        self.assertEqual(len(passed_kwargs["messages"]), 2)
        self.assertEqual(passed_kwargs["messages"][0]["role"], "system")
        self.assertIn("Scaffold: Test bounds.", passed_kwargs["messages"][0]["content"])

    @patch("urllib.request.urlopen")
    def test_wrap_anthropic(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "single_ability": "Scaffold: Test bounds."
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        mock_orig_create = MagicMock(return_value={"content": [{"text": "anthropic ok"}]})
        class MockMessages:
            def __init__(self):
                self.create = mock_orig_create

        class MockAnthropic:
            def __init__(self):
                self.messages = MockMessages()

        fake_anthropic = MockAnthropic()
        ejentum = Ejentum(api_key=self.api_key)
        wrapped = wrap_anthropic(fake_anthropic, ejentum_client=ejentum)

        wrapped.messages.create(
            model="claude-3-7-sonnet",
            max_tokens=500,
            messages=[{"role": "user", "content": "Analyze race condition"}]
        )

        mock_orig_create.assert_called_once()
        passed_kwargs = mock_orig_create.call_args[1]
        self.assertEqual(passed_kwargs["model"], "claude-3-7-sonnet")
        self.assertIn("system", passed_kwargs)
        self.assertIn("Scaffold: Test bounds.", passed_kwargs["system"])

    @patch("urllib.request.urlopen")
    def test_graph_depth_1(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "status": "success",
            "mode": "graph",
            "dag": {
                "root": "CA-001",
                "depth": 1,
                "requires": "TE-004",
                "enhances": "SI-008",
                "fallback": "AB-012"
            },
            "graph_scaffold": "### EJENTUM REASONING GRAPH ACTIVE ###"
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        client = Ejentum(api_key=self.api_key)
        res = client.graph("analyze deadlock", depth=1)

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["dag"]["depth"], 1)
        self.assertEqual(res["dag"]["root"], "CA-001")
        self.assertIn("graph_scaffold", res)

    @patch("urllib.request.urlopen")
    def test_graph_depth_2_multihop(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "status": "success",
            "mode": "graph",
            "version": "1.4.0",
            "dag": {
                "root": "CA-035",
                "depth": 2,
                "requires": "TE-006",
                "enhances": "SI-046",
                "fallback": "AB-036",
                "hop2": {
                    "requires": {"requires": "SP-019", "enhances": "CA-043", "fallback": "SI-017"},
                    "enhances": {"requires": "CA-039", "enhances": "MC-023", "fallback": "TE-007"},
                    "fallback": {"requires": "MC-009", "enhances": "SP-003", "fallback": "MC-047"}
                },
                "execution_plan": [
                    {"stage": 1, "phase": "prerequisite_invariants", "abilities": ["SP-019", "TE-006"]},
                    {"stage": 2, "phase": "root_operator", "abilities": ["CA-035"]},
                    {"stage": 3, "phase": "cognitive_amplification", "abilities": ["SI-046", "MC-023"]},
                    {"stage": 4, "phase": "resiliency_contingency", "abilities": ["AB-036", "MC-047"]}
                ]
            },
            "graph_scaffold": "### EJENTUM RECURSIVE MULTI-HOP COGNITIVE GRAPH (DAG v1.4.0, DEPTH=2) ACTIVE ###"
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        client = Ejentum(api_key=self.api_key)
        res = client.graph("database migration locking table", depth=2)

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["dag"]["depth"], 2)
        self.assertEqual(len(res["dag"]["execution_plan"]), 4)
        self.assertEqual(res["dag"]["execution_plan"][0]["phase"], "prerequisite_invariants")
        self.assertEqual(res["dag"]["execution_plan"][3]["phase"], "resiliency_contingency")

        # Verify sent payload contained depth=2
        sent_req = mock_urlopen.call_args[0][0]
        sent_payload = json.loads(sent_req.data.decode("utf-8"))
        self.assertEqual(sent_payload["depth"], 2)
        self.assertEqual(sent_payload["mode"], "graph")

    @patch("urllib.request.urlopen")
    def test_fts_search(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "status": "success",
            "engine": "sqlite_fts5",
            "query": "deadlock circular dependency",
            "count": 2,
            "results": [
                {
                    "ability_id": "TE-004",
                    "ability_name": "Deadlock Breaker",
                    "domain": "Temporal",
                    "graph_op": "BREAK_DEADLOCK",
                    "rank": -18.966
                },
                {
                    "ability_id": "MC-031",
                    "ability_name": "Contradiction Adjudicator",
                    "domain": "Metacognitive",
                    "graph_op": "ADJUDICATE_DEADLOCK",
                    "rank": -7.513
                }
            ]
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        client = Ejentum(api_key=self.api_key)
        res = client.fts_search("deadlock circular dependency", limit=2)

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["engine"], "sqlite_fts5")
        self.assertEqual(res["count"], 2)
        self.assertEqual(res["results"][0]["ability_id"], "TE-004")

        # Verify GET request URL
        sent_req = mock_urlopen.call_args[0][0]
        self.assertEqual(sent_req.get_method(), "GET")
        self.assertIn("/search/fts?query=deadlock", sent_req.full_url)
        self.assertIn("limit=2", sent_req.full_url)

    def test_fts_search_empty_query_raises(self):
        client = Ejentum(api_key=self.api_key)
        with self.assertRaises(ValueError):
            client.fts_search("")

    @patch("urllib.request.urlopen")
    def test_feedback_telemetry(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "status": "success",
            "recorded": True,
            "ability_id": "CA-001"
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        client = Ejentum(api_key=self.api_key)
        res = client.feedback(ability_id="CA-001", outcome="success", score=0.95)

        self.assertEqual(res["status"], "success")
        self.assertTrue(res["recorded"])


if __name__ == "__main__":
    unittest.main()
