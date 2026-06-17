"""Tests for Fusion orchestration: mock contamination, degradation signalling,
logger hygiene, JSONL integrity, and input validation."""

import json
import logging
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework.fusion_core import Agent, AgentConfig, Fusion, GenerationResult


def make_fusion(num_agents=2, rounds=1, log_file=os.devnull, names=None):
    names = names or [f"A{i}" for i in range(num_agents)]
    agents = [AgentConfig(name=n, model=f"test/model-{i}") for i, n in enumerate(names)]
    return Fusion(
        api_keys={"openrouter": "test-key"},
        agents=agents,
        rounds=rounds,
        max_tokens=50,
        synthesizer=AgentConfig(name="Synth", model="test/synth"),
        log_file=log_file,
    )


class TestMockContamination(unittest.TestCase):
    """A mock/empty agent must not pollute peers' review context or synthesis."""

    def _run_with(self, gen_side_effect):
        calls = []

        def wrapper(self, prompt=None, **kw):
            res = gen_side_effect(self, prompt)
            calls.append({"name": self.name, "prompt": prompt})
            return res

        with patch.object(Agent, "generate", autospec=True, side_effect=wrapper):
            fusion = make_fusion(num_agents=2, rounds=1, names=["A", "B"])
            answer, meta = fusion.debate("q")
        return answer, meta, calls

    def test_mock_agent_excluded_from_peer_review_and_synthesis(self):
        def gen(agent, prompt):
            if agent.name.startswith("Synth"):
                return GenerationResult(content="final synthesis", is_mock=False)
            if agent.name == "B":
                return GenerationResult(content="[MOCK] placeholder from B", is_mock=True)
            return GenerationResult(content="A genuine answer", is_mock=False)

        answer, meta, calls = self._run_with(gen)

        # A's review prompt must not contain B's mock content.
        a_review = next(c["prompt"] for c in calls
                        if c["name"] == "A" and "You will review" in c["prompt"])
        self.assertNotIn("[MOCK]", a_review)
        self.assertNotIn("placeholder from B", a_review)

        # The synthesizer prompt must include A's real answer but not B's mock.
        synth_prompt = next(c["prompt"] for c in calls if c["name"].startswith("Synth"))
        self.assertIn("A genuine answer", synth_prompt)
        self.assertNotIn("placeholder from B", synth_prompt)

    def test_empty_response_excluded_from_synthesis(self):
        def gen(agent, prompt):
            if agent.name.startswith("Synth"):
                return GenerationResult(content="final", is_mock=False)
            if agent.name == "B":
                return GenerationResult(content="", is_mock=False)  # silent empty
            return GenerationResult(content="real content here", is_mock=False)

        _, _, calls = self._run_with(gen)
        synth_prompt = next(c["prompt"] for c in calls if c["name"].startswith("Synth"))
        self.assertIn("real content here", synth_prompt)
        # B contributed an empty block that should be filtered out, so the
        # "[B]" label should not appear with empty content dragging in noise.
        self.assertIn("real content here", synth_prompt)


class TestDegradationFlags(unittest.TestCase):
    """run_meta must carry machine-readable mock/degradation signals."""

    def test_all_mock_sets_flags(self):
        mock_result = GenerationResult(content="[MOCK] x", is_mock=True)
        with patch.object(Agent, "generate", return_value=mock_result):
            fusion = make_fusion(num_agents=2, rounds=1)
            answer, meta = fusion.debate("q")
        self.assertTrue(meta["any_mock"])
        self.assertTrue(meta["is_mock"])
        self.assertIn("agent_is_mock", meta)

    def test_all_real_clears_flags(self):
        real = GenerationResult(content="real answer", is_mock=False)
        with patch.object(Agent, "generate", return_value=real):
            fusion = make_fusion(num_agents=2, rounds=1)
            answer, meta = fusion.debate("q")
        self.assertFalse(meta["any_mock"])
        self.assertFalse(meta["is_mock"])

    def test_partial_mock_flags_any_mock(self):
        def gen(self, prompt=None, **kw):
            if self.name == "A0":
                return GenerationResult(content="[MOCK] x", is_mock=True)
            return GenerationResult(content="real", is_mock=False)

        with patch.object(Agent, "generate", autospec=True, side_effect=gen):
            fusion = make_fusion(num_agents=2, rounds=1, names=["A0", "A1"])
            _, meta = fusion.debate("q")
        self.assertTrue(meta["any_mock"])

    def test_debate_still_returns_when_all_degraded(self):
        mock_result = GenerationResult(content="[MOCK SYNTHESIS] x", is_mock=True)
        with patch.object(Agent, "generate", return_value=mock_result):
            fusion = make_fusion(num_agents=2, rounds=2)
            answer, meta = fusion.debate("q")
        self.assertIsInstance(answer, str)
        self.assertTrue(meta["any_mock"])


class TestInputValidation(unittest.TestCase):
    def test_empty_agents_raises(self):
        with self.assertRaises(ValueError):
            Fusion(api_keys={"openrouter": "k"}, agents=[], log_file=os.devnull)

    def test_duplicate_agent_names_raises(self):
        agents = [AgentConfig(name="Dup", model="m1"), AgentConfig(name="Dup", model="m2")]
        with self.assertRaises(ValueError):
            Fusion(api_keys={"openrouter": "k"}, agents=agents, log_file=os.devnull)

    def test_unique_names_ok(self):
        agents = [AgentConfig(name="X", model="m1"), AgentConfig(name="Y", model="m2")]
        fusion = Fusion(api_keys={"openrouter": "k"}, agents=agents, log_file=os.devnull)
        self.assertEqual(len(fusion.agents), 2)


class TestLoggerHygiene(unittest.TestCase):
    def test_per_instance_logger_no_filehandler_no_accumulation(self):
        f1 = make_fusion()
        f2 = make_fusion()
        # Distinct loggers, no shared global handler accumulation.
        self.assertNotEqual(f1.logger.name, f2.logger.name)
        for f in (f1, f2):
            self.assertEqual(len(f.logger.handlers), 1)
            handler = f.logger.handlers[0]
            self.assertIsInstance(handler, logging.StreamHandler)
            self.assertNotIsInstance(handler, logging.FileHandler)
            # Logs go to stderr, not stdout (which carries the progress UI).
            self.assertIs(handler.stream, sys.stderr)

    def test_jsonl_run_log_is_valid_jsonl(self):
        real = GenerationResult(content="answer", usage={}, raw_response={"ok": True})
        with tempfile.NamedTemporaryFile("r", suffix=".jsonl", delete=False) as tf:
            path = tf.name
        try:
            with patch.object(Agent, "generate", return_value=real):
                fusion = make_fusion(num_agents=2, rounds=1, log_file=path)
                fusion.debate("q")
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln for ln in f.read().splitlines() if ln.strip()]
            self.assertGreater(len(lines), 0)
            for ln in lines:
                json.loads(ln)  # must not raise — file is pure JSONL
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
