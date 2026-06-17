"""Tests for the Agent HTTP/transport, response-parsing, retry and fallback
paths — the parts that actually break against real provider responses.

These deliberately exercise the urllib transport (by forcing ``session=None``)
so the real ``_post`` body, header handling, and JSON decoding are covered,
rather than mocking ``_post`` away.
"""

import io
import json
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework.fusion_core import Agent, GenerationResult


class FakeResp:
    """Minimal stand-in for the urlopen context-manager response."""

    def __init__(self, body: str, status: int = 200):
        self._b = body.encode("utf-8")
        self.status = status

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def make_urlopen(handler):
    """Build a fake urlopen that records requests and delegates to ``handler``.

    ``handler(url, headers, body) -> (status, text)`` for 2xx, or may raise an
    ``urllib.error.HTTPError`` to simulate a non-2xx response.
    """
    import urllib.error

    captured = []

    def fake_urlopen(req, timeout=None):
        url = req.full_url
        headers = dict(req.headers)
        body = req.data.decode("utf-8") if req.data else ""
        captured.append({"url": url, "headers": headers, "body": body})
        result = handler(url, headers, body)
        if isinstance(result, Exception):
            raise result
        status, text = result
        return FakeResp(text, status)

    fake_urlopen.captured = captured
    return fake_urlopen


class AgentHTTPTestBase(unittest.TestCase):
    def _agent(self, model="openrouter/model", **kwargs):
        defaults = dict(
            name="A",
            model=model,
            api_keys={"openrouter": "or-key", "gemini": "gem-key"},
            max_retries=1,
            enable_mock_fallback=False,
        )
        defaults.update(kwargs)
        agent = Agent(**defaults)
        agent.session = None  # force the urllib transport for deterministic tests
        return agent

    def setUp(self):
        # Don't actually sleep during retry/backoff tests.
        patcher = patch("framework.fusion_core.time.sleep", return_value=None)
        self.addCleanup(patcher.stop)
        patcher.start()


class TestResponseParsing(AgentHTTPTestBase):
    """Malformed/edge-case provider responses must not crash or silently lie."""

    def test_null_content_coerced_to_empty_string(self):
        body = json.dumps({"choices": [{"message": {"content": None}}], "usage": {}})
        with patch("urllib.request.urlopen", make_urlopen(lambda *a: (200, body))):
            res = self._agent().generate("hi")
        self.assertEqual(res.content, "")
        self.assertIsNotNone(res.content)  # never None — downstream .strip() is safe
        res.content.strip()  # would AttributeError on None

    def test_empty_choices_list_does_not_crash(self):
        body = json.dumps({"choices": []})
        with patch("urllib.request.urlopen", make_urlopen(lambda *a: (200, body))):
            res = self._agent(enable_mock_fallback=True).generate("hi")
        # Empty choices is a failed call → mock fallback, not an IndexError.
        self.assertTrue(res.is_mock)

    def test_empty_choices_advances_to_fallback_model(self):
        good = json.dumps({"choices": [{"message": {"content": "from fallback"}}], "usage": {}})

        def handler(url, headers, body):
            model = json.loads(body)["model"]
            if model == "openrouter/primary":
                return (200, json.dumps({"choices": []}))
            return (200, good)

        agent = self._agent(model="openrouter/primary", fallback_models=["openrouter/backup"])
        with patch("urllib.request.urlopen", make_urlopen(handler)):
            res = agent.generate("hi")
        self.assertEqual(res.content, "from fallback")
        self.assertFalse(res.is_mock)

    def test_200_with_error_body_is_treated_as_failure(self):
        body = json.dumps({"error": {"message": "upstream exploded", "code": 502}})
        with patch("urllib.request.urlopen", make_urlopen(lambda *a: (200, body))):
            res = self._agent().generate("hi")
        # No choices → empty content with an error recorded, NOT a fake success.
        self.assertEqual(res.content, "")
        self.assertIsNotNone(res.error)

    def test_gemini_empty_candidates_does_not_crash(self):
        body = json.dumps({"promptFeedback": {"blockReason": "SAFETY"}})
        agent = self._agent(model="gemini-2.0-flash", api_keys={"gemini": "g"},
                            enable_mock_fallback=True)
        with patch("urllib.request.urlopen", make_urlopen(lambda *a: (200, body))):
            res = agent.generate("hi")
        self.assertTrue(res.is_mock)  # safety block → mock fallback, not IndexError

    def test_gemini_candidate_without_parts_does_not_crash(self):
        body = json.dumps({"candidates": [{"finishReason": "RECITATION", "content": {}}]})
        agent = self._agent(model="gemini-2.0-flash", api_keys={"gemini": "g"},
                            enable_mock_fallback=False)
        with patch("urllib.request.urlopen", make_urlopen(lambda *a: (200, body))):
            res = agent.generate("hi")
        # No parts → treated as failed (empty content + error), not a crash.
        self.assertEqual(res.content, "")
        self.assertIsNotNone(res.error)

    def test_gemini_happy_path(self):
        body = json.dumps({"candidates": [{"content": {"parts": [{"text": "hello"}]}}]})
        agent = self._agent(model="gemini-2.0-flash", api_keys={"gemini": "g"})
        with patch("urllib.request.urlopen", make_urlopen(lambda *a: (200, body))):
            res = agent.generate("hi")
        self.assertEqual(res.content, "hello")

    def test_non_json_2xx_body_does_not_crash(self):
        # A proxy/HTML error page returned with HTTP 200.
        with patch("urllib.request.urlopen", make_urlopen(lambda *a: (200, "<html>oops</html>"))):
            res = self._agent().generate("hi")
        self.assertEqual(res.content, "")
        self.assertIsNotNone(res.error)


class TestHeaderIsolation(AgentHTTPTestBase):
    """The OpenRouter bearer token must never leak to another provider."""

    def test_post_does_not_mutate_caller_headers(self):
        agent = self._agent(model="openrouter/model")
        shared = {"Content-Type": "application/json"}
        with patch("urllib.request.urlopen", make_urlopen(lambda *a: (200, '{"choices":[{"message":{"content":"x"}}]}'))):
            agent._post("openrouter/model", shared, {"messages": [{"role": "user", "content": "hi"}],
                                                     "max_tokens": 10, "temperature": 0.5})
        self.assertNotIn("Authorization", shared)

    def test_openrouter_token_not_sent_to_gemini_fallback(self):
        import urllib.error

        def handler(url, headers, body):
            if "openrouter" in url:
                return urllib.error.HTTPError(url, 500, "boom", {}, io.BytesIO(b"boom"))
            return (200, json.dumps({"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}))

        fake = make_urlopen(handler)
        agent = self._agent(model="openrouter/model",
                            api_keys={"openrouter": "SECRET-OR", "gemini": "gk"},
                            fallback_models=["gemini-2.0-flash"])
        with patch("urllib.request.urlopen", fake):
            agent.generate("hi")

        gemini_reqs = [c for c in fake.captured if "generativelanguage" in c["url"]]
        self.assertTrue(gemini_reqs)
        for req in gemini_reqs:
            auth = req["headers"].get("Authorization")
            self.assertNotIn("SECRET-OR", auth or "")
            self.assertIsNone(auth)

    def test_gemini_key_url_encoded(self):
        captured = {}

        def handler(url, headers, body):
            captured["url"] = url
            return (200, json.dumps({"candidates": [{"content": {"parts": [{"text": "x"}]}}]}))

        agent = self._agent(model="gemini-2.0-flash", api_keys={"gemini": "key/with+special=chars"})
        with patch("urllib.request.urlopen", make_urlopen(handler)):
            agent.generate("hi")
        # Raw special chars must be percent-encoded in the query string.
        self.assertNotIn("key/with+special=chars", captured["url"])
        self.assertIn("key%2Fwith%2Bspecial%3Dchars", captured["url"])


class TestRetryAndFallback(AgentHTTPTestBase):
    def test_5xx_retries_then_advances_to_fallback(self):
        import urllib.error
        calls = {"primary": 0, "backup": 0}

        def handler(url, headers, body):
            model = json.loads(body)["model"]
            if model == "p/primary":
                calls["primary"] += 1
                return urllib.error.HTTPError(url, 503, "down", {}, io.BytesIO(b"down"))
            calls["backup"] += 1
            return (200, json.dumps({"choices": [{"message": {"content": "ok"}}]}))

        agent = self._agent(model="p/primary", fallback_models=["p/backup"], max_retries=3)
        with patch("urllib.request.urlopen", make_urlopen(handler)):
            res = agent.generate("hi")
        self.assertEqual(res.content, "ok")
        self.assertEqual(calls["primary"], 3)  # exhausted retries on primary
        self.assertEqual(calls["backup"], 1)   # then advanced to fallback

    def test_generic_4xx_advances_to_fallback(self):
        import urllib.error

        def handler(url, headers, body):
            model = json.loads(body)["model"]
            if model == "p/primary":
                # 402 insufficient credits — non-retryable, but should try fallback.
                return urllib.error.HTTPError(url, 402, "no credits", {}, io.BytesIO(b'{"error":"payment"}'))
            return (200, json.dumps({"choices": [{"message": {"content": "rescued"}}]}))

        agent = self._agent(model="p/primary", fallback_models=["p/backup"])
        with patch("urllib.request.urlopen", make_urlopen(handler)):
            res = agent.generate("hi")
        self.assertEqual(res.content, "rescued")

    def test_403_with_mock_disabled_reports_error(self):
        import urllib.error
        with patch("urllib.request.urlopen",
                   make_urlopen(lambda *a: urllib.error.HTTPError("u", 403, "no", {}, io.BytesIO(b"forbidden")))):
            res = self._agent(enable_mock_fallback=False).generate("hi")
        self.assertEqual(res.content, "")
        self.assertIn("403", res.error)


class TestMockGate(AgentHTTPTestBase):
    def test_any_usable_key_considers_fallback_models(self):
        # Primary routes to openrouter (no key), fallback is native gemini (has key).
        agent = Agent(name="A", model="openrouter/model",
                      api_keys={"gemini": "g"},
                      fallback_models=["gemini-2.0-flash"],
                      enable_mock_fallback=True)
        self.assertTrue(agent._any_usable_key())

    def test_no_keys_at_all_short_circuits_to_mock(self):
        agent = Agent(name="A", model="openrouter/model", api_keys={}, enable_mock_fallback=True)
        res = agent.generate("hi")
        self.assertTrue(res.is_mock)
        self.assertIsNone(res.error)

    def test_missing_primary_key_still_tries_fallback_provider(self):
        body = json.dumps({"candidates": [{"content": {"parts": [{"text": "gemini ok"}]}}]})
        agent = Agent(name="A", model="openrouter/model",
                      api_keys={"gemini": "g"},
                      fallback_models=["gemini-2.0-flash"],
                      max_retries=1, enable_mock_fallback=True)
        agent.session = None
        with patch("urllib.request.urlopen", make_urlopen(lambda *a: (200, body))):
            res = agent.generate("hi")
        # Should reach the gemini fallback, not return a mock for the missing OR key.
        self.assertFalse(res.is_mock)
        self.assertEqual(res.content, "gemini ok")


class TestGeminiPayload(AgentHTTPTestBase):
    def test_generation_config_included(self):
        captured = {}

        def handler(url, headers, body):
            captured["body"] = json.loads(body)
            return (200, json.dumps({"candidates": [{"content": {"parts": [{"text": "x"}]}}]}))

        agent = self._agent(model="gemini-2.0-flash", api_keys={"gemini": "g"})
        with patch("urllib.request.urlopen", make_urlopen(handler)):
            agent.generate("hi", max_tokens=123, temperature=0.4)
        self.assertIn("generationConfig", captured["body"])
        self.assertEqual(captured["body"]["generationConfig"]["maxOutputTokens"], 123)
        self.assertEqual(captured["body"]["generationConfig"]["temperature"], 0.4)
        self.assertIn("contents", captured["body"])


if __name__ == "__main__":
    unittest.main()
