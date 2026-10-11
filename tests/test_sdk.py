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


if __name__ == "__main__":
    unittest.main()
