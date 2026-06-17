"""Tests for the CLI config layer: api_keys deep-merge, AgentConfig validation,
.env shell-safety, and onboarding robustness."""

import os
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

# Make both the framework and the CLI module importable.
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "chat-cli", "cli"))

import main as cli  # noqa: E402
from framework.fusion_core import AgentConfig  # noqa: E402


class TestMergeApiKeys(unittest.TestCase):
    def test_partial_overlay_does_not_drop_existing(self):
        base = {"openrouter": "or-key", "gemini": ""}
        merged = cli.merge_api_keys(base, {"gemini": "g"})
        self.assertEqual(merged["openrouter"], "or-key")  # preserved
        self.assertEqual(merged["gemini"], "g")           # overridden

    def test_empty_overlay_values_ignored(self):
        base = {"openrouter": "or-key"}
        merged = cli.merge_api_keys(base, {"openrouter": "", "gemini": None})
        self.assertEqual(merged["openrouter"], "or-key")

    def test_none_overlay(self):
        base = {"openrouter": "or-key"}
        self.assertEqual(cli.merge_api_keys(base, None), {"openrouter": "or-key"})


class TestAgentConfigFromDict(unittest.TestCase):
    def test_valid(self):
        cfg = cli.agent_config_from_dict({"name": "A", "model": "m", "role": "r"})
        self.assertIsInstance(cfg, AgentConfig)
        self.assertEqual(cfg.name, "A")

    def test_unknown_field_raises(self):
        with self.assertRaises(ValueError) as ctx:
            cli.agent_config_from_dict({"name": "A", "model": "m", "temperature": 0.5})
        self.assertIn("temperature", str(ctx.exception))

    def test_missing_required_raises(self):
        with self.assertRaises(ValueError):
            cli.agent_config_from_dict({"name": "A"})


class TestBuildFusionFromConfig(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ("OPENROUTER_API_KEY", "GEMINI_API_KEY")}
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ.pop("GEMINI_API_KEY", None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_env_key_survives_partial_config_keys(self):
        os.environ["OPENROUTER_API_KEY"] = "env-or"
        # User config only supplies a gemini key — must NOT wipe the env OR key.
        fusion = cli.build_fusion_from_config({"api_keys": {"gemini": "cfg-gemini"}, "rounds": 1})
        self.assertEqual(fusion.api_keys.get("openrouter"), "env-or")
        self.assertEqual(fusion.api_keys.get("gemini"), "cfg-gemini")

    def test_no_keys_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            cli.build_fusion_from_config({"rounds": 1})

    def test_bad_agent_config_raises_value_error(self):
        os.environ["OPENROUTER_API_KEY"] = "env-or"
        with self.assertRaises(ValueError):
            cli.build_fusion_from_config({"agents": [{"name": "A", "model": "m", "bogus": 1}]})


class TestEnvFileSafety(unittest.TestCase):
    def test_dangerous_key_is_shell_quoted(self):
        evil_key = 'abc$(rm -rf ~)`whoami`"quote'
        with tempfile.TemporaryDirectory() as d:
            env_path = os.path.join(d, ".env")
            with patch("builtins.input", return_value="y"):
                cli.maybe_save_env_vars({"openrouter": evil_key}, env_path=env_path)
            with open(env_path, "r", encoding="utf-8") as f:
                content = f.read()
        env_var = cli.API_CONFIG["openrouter"]["api_key_env"]
        expected = f"export {env_var}={shlex.quote(evil_key)}\n"
        self.assertEqual(content, expected)
        # The raw, unquoted dangerous form must not appear.
        self.assertNotIn('="abc$(rm', content)

    def test_decline_save_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            env_path = os.path.join(d, ".env")
            with patch("builtins.input", return_value="n"):
                cli.maybe_save_env_vars({"openrouter": "k"}, env_path=env_path)
            self.assertFalse(os.path.exists(env_path))


class TestOnboardingRobustness(unittest.TestCase):
    def test_non_interactive_stdin_does_not_crash(self):
        # getpass fails (no tty) and input hits EOFError (piped/empty stdin).
        with patch("getpass.getpass", side_effect=Exception("no tty")), \
             patch("builtins.input", side_effect=EOFError):
            keys = cli.prompt_for_api_keys_interactive()
        self.assertEqual(keys, {})


if __name__ == "__main__":
    unittest.main()
