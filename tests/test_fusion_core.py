"""Tests for the FUSION framework core module."""

import json
import unittest
from unittest.mock import MagicMock, patch

import sys
import os

# Ensure the framework package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework.fusion_core import Agent, AgentConfig, Fusion, GenerationResult


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------

class TestAgentGetProvider(unittest.TestCase):
    """Tests for Agent._get_provider routing logic."""

    def _make_agent(self, model="test-model", api_keys=None):
        keys = api_keys or {"openrouter": "or-key", "gemini": "gem-key"}
        return Agent(name="test", model=model, api_keys=keys)

    def test_bare_gemini_routes_to_gemini(self):
        agent = self._make_agent()
        self.assertEqual(agent._get_provider("gemini-2.0-flash"), "gemini")

    def test_slash_prefixed_gemini_routes_to_openrouter(self):
        """google/gemini-... should go through OpenRouter, not native Gemini."""
        agent = self._make_agent()
        self.assertEqual(agent._get_provider("google/gemini-2.5-pro-preview"), "openrouter")

    def test_bare_gemini_without_key_routes_to_openrouter(self):
        agent = self._make_agent(api_keys={"openrouter": "or-key"})
        self.assertEqual(agent._get_provider("gemini-2.0-flash"), "openrouter")

    def test_non_gemini_routes_to_openrouter(self):
        agent = self._make_agent()
        self.assertEqual(agent._get_provider("anthropic/claude-3.5-sonnet"), "openrouter")

    def test_bare_non_gemini_routes_to_openrouter(self):
        agent = self._make_agent()
        self.assertEqual(agent._get_provider("llama-3-70b"), "openrouter")


class TestAgentGenerate(unittest.TestCase):
    """Tests for Agent.generate with mocked HTTP calls."""

    def _make_agent(self, model="test/model", **kwargs):
        defaults = dict(
            name="TestAgent",
            model=model,
            api_keys={"openrouter": "test-key"},
            max_retries=1,
            enable_mock_fallback=True,
        )
        defaults.update(kwargs)
        return Agent(**defaults)

    def test_mock_fallback_when_no_api_key(self):
        agent = self._make_agent(api_keys={})
        result = agent.generate("Hello")
        self.assertTrue(result.is_mock)
        self.assertIn("[MOCK]", result.content)
        self.assertIsNone(result.error)

    @patch.object(Agent, "_post")
    def test_successful_openrouter_response(self, mock_post):
        mock_post.return_value = (
            200,
            '{"choices":[{"message":{"content":"Test response"}}],"usage":{}}',
            {"choices": [{"message": {"content": "Test response"}}], "usage": {}},
        )
        agent = self._make_agent()
        result = agent.generate("Hello")
        self.assertFalse(result.is_mock)
        self.assertEqual(result.content, "Test response")
        self.assertIsNone(result.error)

    @patch.object(Agent, "_post")
    def test_successful_gemini_response(self, mock_post):
        mock_post.return_value = (
            200,
            '{}',
            {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "Gemini says hi"}]
                    }
                }]
            },
        )
        agent = self._make_agent(
            model="gemini-2.0-flash",
            api_keys={"gemini": "gem-key"},
        )
        result = agent.generate("Hello")
        self.assertEqual(result.content, "Gemini says hi")

    @patch.object(Agent, "_post")
    def test_http_error_returns_error(self, mock_post):
        mock_post.return_value = (403, '{"error":"forbidden"}', {})
        agent = self._make_agent(enable_mock_fallback=False)
        result = agent.generate("Hello")
        self.assertIsNotNone(result.error)
        self.assertIn("403", result.error)

    @patch.object(Agent, "_post")
    def test_retries_on_server_error(self, mock_post):
        """500 errors should trigger retries then mock fallback."""
        mock_post.side_effect = RuntimeError("HTTP 500: server error")
        agent = self._make_agent(max_retries=2)
        # With mock fallback enabled, should get a mock response
        result = agent.generate("Hello")
        self.assertTrue(result.is_mock)
        # _post should have been called max_retries times
        self.assertEqual(mock_post.call_count, 2)

    @patch.object(Agent, "_post")
    def test_system_prompt_included(self, mock_post):
        """System prompt should appear in the messages payload."""
        mock_post.return_value = (
            200, '{}',
            {"choices": [{"message": {"content": "ok"}}], "usage": {}},
        )
        agent = self._make_agent(default_system_prompt="Be helpful")
        agent.generate("Hello")
        # Inspect what payload was sent
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["payload"] if "payload" in (call_kwargs[1] or {}) else call_kwargs[0][2]
        messages = payload["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "Be helpful")

    @patch.object(Agent, "_post")
    def test_extra_messages_included(self, mock_post):
        mock_post.return_value = (
            200, '{}',
            {"choices": [{"message": {"content": "ok"}}], "usage": {}},
        )
        agent = self._make_agent()
        agent.generate("Hello", extra_messages=[{"role": "user", "content": "context"}])
        # _post is called with keyword args: model_name, headers, payload
        payload = mock_post.call_args[1]["payload"]
        roles = [m["role"] for m in payload["messages"]]
        self.assertIn("user", roles)
        # Should have context message + actual prompt
        self.assertEqual(len(payload["messages"]), 2)

    @patch.object(Agent, "_post")
    def test_model_unavailable_tries_fallback(self, mock_post):
        """When primary model returns 404 'not a valid model id', try fallback."""
        call_count = {"n": 0}

        def side_effect(model_name, headers, payload):
            call_count["n"] += 1
            if model_name == "bad/model":
                return 404, '{"error":"not a valid model id"}', {}
            return (
                200, '{}',
                {"choices": [{"message": {"content": "fallback ok"}}], "usage": {}},
            )

        mock_post.side_effect = side_effect
        agent = self._make_agent(
            model="bad/model",
            fallback_models=["good/model"],
            enable_mock_fallback=False,
        )
        result = agent.generate("Hello")
        self.assertEqual(result.content, "fallback ok")
        self.assertEqual(call_count["n"], 2)


# ---------------------------------------------------------------------------
# Fusion tests
# ---------------------------------------------------------------------------

class TestFusionDebate(unittest.TestCase):
    """Tests for Fusion.debate orchestration with mocked agents."""

    def _make_fusion(self, rounds=1, num_agents=2):
        agents = [
            AgentConfig(name=f"Agent{i}", model=f"test/model-{i}")
            for i in range(num_agents)
        ]
        return Fusion(
            api_keys={"openrouter": "test-key"},
            agents=agents,
            rounds=rounds,
            max_tokens=100,
            log_file=os.devnull,
        )

    @patch.object(Agent, "generate")
    def test_debate_returns_string_and_meta(self, mock_generate):
        mock_generate.return_value = GenerationResult(
            content="Test answer", usage={}, raw_response={}
        )
        fusion = self._make_fusion(rounds=1, num_agents=2)
        answer, meta = fusion.debate("What is 2+2?")
        self.assertIsInstance(answer, str)
        self.assertIn("final_answer", meta)
        self.assertEqual(answer, "Test answer")

    @patch.object(Agent, "generate")
    def test_debate_calls_agents_correct_number_of_times(self, mock_generate):
        mock_generate.return_value = GenerationResult(
            content="Response", usage={}, raw_response={}
        )
        num_agents = 3
        rounds = 2
        fusion = self._make_fusion(rounds=rounds, num_agents=num_agents)
        fusion.debate("test query")
        # Expected calls: initial (3) + review rounds (3*2) + synthesis (1) = 10
        expected = num_agents + (num_agents * rounds) + 1
        self.assertEqual(mock_generate.call_count, expected)

    @patch.object(Agent, "generate")
    def test_debate_paper_mode(self, mock_generate):
        mock_generate.return_value = GenerationResult(
            content="Paper response", usage={}, raw_response={}
        )
        fusion = self._make_fusion(rounds=1, num_agents=2)
        answer, meta = fusion.debate("Write about AI", paper_mode=True)
        self.assertIsInstance(answer, str)

    @patch.object(Agent, "generate")
    def test_debate_progress_callback(self, mock_generate):
        mock_generate.return_value = GenerationResult(
            content="Response", usage={}, raw_response={}
        )
        events = []
        fusion = self._make_fusion(rounds=1, num_agents=2)
        fusion.debate("test", progress_callback=lambda e: events.append(e))
        # Should have received multiple progress events
        self.assertGreater(len(events), 0)
        event_types = {e.get("type") for e in events}
        self.assertIn("status", event_types)
        self.assertIn("spinner_start", event_types)


class TestFusionHelpers(unittest.TestCase):
    """Tests for Fusion static helper methods."""

    def test_extract_refined_with_marker(self):
        text = "Critique: blah\nRefined Answer: This is the good part."
        result = Fusion._extract_refined(text)
        self.assertEqual(result, "This is the good part.")

    def test_extract_refined_without_marker(self):
        text = "Just a plain response."
        result = Fusion._extract_refined(text)
        self.assertEqual(result, "Just a plain response.")

    def test_extract_refined_mock_response(self):
        text = "[MOCK] This is a mock response."
        result = Fusion._extract_refined(text)
        self.assertEqual(result, "[MOCK] This is a mock response.")

    def test_extract_refined_mock_synthesis_response(self):
        text = "[MOCK SYNTHESIS] placeholder synthesis."
        result = Fusion._extract_refined(text)
        self.assertEqual(result, "[MOCK SYNTHESIS] placeholder synthesis.")

    def test_extract_refined_ignores_in_critique_mention(self):
        """An in-sentence 'refined answer:' must not be mistaken for the header."""
        text = (
            "Critique: I will now give my refined answer: it was weak.\n"
            "Refined Answer: The real refined content."
        )
        result = Fusion._extract_refined(text)
        self.assertEqual(result, "The real refined content.")

    def test_extract_refined_markdown_header(self):
        text = "Critique: blah\n**Refined Answer:** the markdown answer"
        result = Fusion._extract_refined(text)
        self.assertEqual(result, "the markdown answer")

    def test_extract_refined_bulleted_header(self):
        text = "- Critique: blah\n- Refined Answer: bulleted answer"
        result = Fusion._extract_refined(text)
        self.assertEqual(result, "bulleted answer")

    def test_extract_refined_empty_string(self):
        self.assertEqual(Fusion._extract_refined(""), "")

    def test_build_initial_prompt_normal(self):
        prompt = Fusion._build_initial_prompt("What is AI?", paper_mode=False)
        self.assertIn("What is AI?", prompt)
        self.assertNotIn("Paper Writing", prompt)

    def test_build_initial_prompt_paper_mode(self):
        prompt = Fusion._build_initial_prompt("What is AI?", paper_mode=True)
        self.assertIn("Paper Writing Mode", prompt)

    def test_build_review_prompt_includes_others(self):
        prompt = Fusion._build_review_prompt(
            query="test",
            self_response="my answer",
            other_responses={"AgentB": "their answer"},
            paper_mode=False,
        )
        self.assertIn("AgentB", prompt)
        self.assertIn("their answer", prompt)

    def test_build_synthesis_prompt(self):
        prompt = Fusion._build_synthesis_prompt(
            query="test",
            agent_outputs={"A": "answer1", "B": "answer2"},
            paper_mode=False,
        )
        self.assertIn("answer1", prompt)
        self.assertIn("answer2", prompt)
        self.assertIn("synthesizer", prompt.lower())

    def test_estimate_tokens(self):
        self.assertEqual(Fusion._estimate_tokens("abcd"), 1)
        self.assertEqual(Fusion._estimate_tokens("a" * 400), 100)

    def test_build_system_prompt(self):
        cfg = AgentConfig(name="Expert", model="m", role="mathematics")
        prompt = Fusion._build_system_prompt(cfg)
        self.assertIn("Expert", prompt)
        self.assertIn("mathematics", prompt)


if __name__ == "__main__":
    unittest.main()
