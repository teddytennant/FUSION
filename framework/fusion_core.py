#!/usr/bin/env python3
"""
FUSION (Federated Unified Systems Integration Orchestration Network) - Core Framework

This module contains the core classes for the FUSION framework, including:
- Agent: A wrapper for language model APIs.
- Fusion: The orchestrator for multi-agent debates.
"""

import json
import os
import re
import sys
import time
import logging
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

# Optional torch import for advanced features (e.g., cosine similarity)
try:
    import torch  # noqa: F401
except Exception:
    torch = None  # type: ignore

# HTTP client: prefer requests if available, else fall back to the
# always-imported urllib (above).
try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore


# --- Constants ---
DEFAULT_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
DEFAULT_RUN_LOG = os.path.join(DEFAULT_LOG_DIR, "runs.jsonl")

# --- API Configuration ---
# Using a dictionary to allow for easy expansion to other providers
API_CONFIG = {
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "api_key_env": "GEMINI_API_KEY",
    },
    # Add other providers here, e.g., Anthropic, OpenAI directly
}


@dataclass
class AgentConfig:
    """Declarative configuration for an agent."""
    name: str
    model: str
    role: Optional[str] = None
    fallback_models: Optional[List[str]] = None


@dataclass
class GenerationResult:
    """Normalized result object for model generations."""
    content: str
    usage: Dict[str, Any] = field(default_factory=dict)
    raw_response: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    is_mock: bool = False


class Agent:
    """Wrapper for various LLM APIs, handling provider-specific logic."""

    def __init__(
        self,
        name: str,
        model: str,
        api_keys: Dict[str, str],
        max_retries: int = 3,
        timeout: int = 60,
        request_headers: Optional[Dict[str, str]] = None,
        default_system_prompt: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        enable_mock_fallback: bool = True,
    ) -> None:
        self.name = name
        self.model = model
        self.api_keys = api_keys
        self.max_retries = max_retries
        self.timeout = timeout
        self.request_headers = request_headers or {}
        self.default_system_prompt = default_system_prompt
        self.fallback_models = list(fallback_models or [])
        self.enable_mock_fallback = enable_mock_fallback
        self.session = requests.Session() if requests else None

    def _get_provider(self, model_name: str) -> str:
        """Determine the provider from the model name.

        Models accessed via OpenRouter use slash-prefixed IDs like
        ``google/gemini-2.5-pro-preview``.  We only route to the
        native Gemini API when a bare model name (no slash) is used
        *and* a Gemini API key is available.
        """
        if "/" not in model_name and "gemini" in model_name.lower() and self._get_api_key("gemini"):
            return "gemini"
        # Default to openrouter for any slash-prefixed or non-gemini model
        return "openrouter"

    def _get_api_key(self, provider: str) -> Optional[str]:
        """Retrieve the API key for a given provider."""
        return self.api_keys.get(provider)

    def _any_usable_key(self) -> bool:
        """True if at least one candidate model (primary or fallback) has a key.

        The mock short-circuit must consider fallback models, not just the
        primary, otherwise a missing primary key would mask a perfectly good
        fallback that routes to a different provider.
        """
        for model_name in [self.model] + self.fallback_models:
            key = self._get_api_key(self._get_provider(model_name))
            if key and key.strip():
                return True
        return False

    def _post(self, model_name: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Tuple[int, str, Dict[str, Any]]:
        """Send a POST request, adapting to the provider's API format."""
        provider = self._get_provider(model_name)
        config = API_CONFIG.get(provider)
        if not config:
            raise ValueError(f"Provider '{provider}' not configured.")

        api_key = self._get_api_key(provider)
        if not api_key:
            return 401, '{"error": "API key not found"}', {}

        # Work on a per-request copy so an Authorization header for one provider
        # is never carried over to a later (e.g. fallback) request to a
        # different provider. Mutating the caller's shared dict would leak the
        # OpenRouter bearer token into native-Gemini requests.
        req_headers = dict(headers)
        req_headers.pop("Authorization", None)

        url = config["url"]
        if provider == "gemini":
            url = url.format(model=urllib.parse.quote(model_name, safe="")) + f"?key={urllib.parse.quote(api_key, safe='')}"
            # Gemini API expects a list of content objects, one per message.
            # System messages are mapped to role "model" as Gemini doesn't
            # support a dedicated system role in the contents array.
            role_map = {"system": "model", "assistant": "model", "user": "user"}
            gemini_contents = []
            for m in payload["messages"]:
                role = role_map.get(m["role"], "user")
                gemini_contents.append({
                    "role": role,
                    "parts": [{"text": m["content"]}],
                })
            gemini_payload: Dict[str, Any] = {"contents": gemini_contents}
            gen_config: Dict[str, Any] = {}
            if payload.get("max_tokens") is not None:
                gen_config["maxOutputTokens"] = payload["max_tokens"]
            if payload.get("temperature") is not None:
                gen_config["temperature"] = payload["temperature"]
            if gen_config:
                gemini_payload["generationConfig"] = gen_config
            payload = gemini_payload
        else:  # openrouter and others
            req_headers["Authorization"] = f"Bearer {api_key}"

        if requests and self.session:
            resp = self.session.post(url, headers=req_headers, json=payload, timeout=self.timeout)
            data: Dict[str, Any] = {}
            if resp.ok:
                try:
                    data = resp.json()
                except Exception:
                    pass
            return resp.status_code, resp.text, data

        # Fallback to urllib
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=req_headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                text = r.read().decode("utf-8")
                try:
                    data = json.loads(text)
                except Exception:
                    data = {}
                return r.status, text, data
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return e.code, body, {}

    @staticmethod
    def _build_mock_response(prompt: str, agent_name: str) -> str:
        summary = prompt.strip().split("\n")[0]
        summary = (summary[:140] + "…") if len(summary) > 140 else summary
        
        # Special handling for synthesizer
        if "synthesizer" in agent_name.lower():
            return (
                f"[MOCK SYNTHESIS] This is a placeholder synthesis because the API key is missing or unavailable.\n"
                f"Original query: {summary}\n"
                "To get a real synthesized answer, provide a valid OPENROUTER_API_KEY with access to the requested models.\n"
                "The synthesis would normally combine insights from all participating agents into a comprehensive response."
            )
        
        return (
            f"[MOCK] This is a placeholder response from {agent_name} because the API key is missing or unavailable.\n"
            f"Prompt snippet: {summary}\n"
            "Provide a valid OPENROUTER_API_KEY (or enable this model) to get real outputs."
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        extra_messages: Optional[List[Dict[str, str]]] = None,
    ) -> GenerationResult:
        """Call the model to generate a response to the user prompt."""
        messages: List[Dict[str, str]] = []
        effective_system_prompt = system_prompt or self.default_system_prompt
        if effective_system_prompt:
            messages.append({"role": "system", "content": effective_system_prompt})
        if extra_messages:
            messages.extend(extra_messages)
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Content-Type": "application/json",
            **self.request_headers,
        }

        if self.enable_mock_fallback and not self._any_usable_key():
            mock_text = self._build_mock_response(prompt, self.name)
            return GenerationResult(
                content=mock_text,
                usage={},
                raw_response={"mock": True, "reason": "missing_api_key", "model": self.model},
                error=None,
                is_mock=True,
            )

        models_to_try: List[str] = [self.model] + self.fallback_models
        last_err: Optional[str] = None
        for model_name in models_to_try:
            payload: Dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if seed is not None:
                payload["seed"] = seed

            backoff = 1.0
            for attempt in range(1, self.max_retries + 1):
                try:
                    status, text, data = self._post(model_name=model_name, headers=headers, payload=payload)
                    if status >= 400:
                        # Retryable transient errors: raise to trigger backoff.
                        if status in (408, 409, 429) or 500 <= status < 600:
                            raise RuntimeError(f"HTTP {status}: {text}")
                        # Any other 4xx is non-retryable for this model. Record
                        # the reason and advance to the next fallback model
                        # rather than aborting the whole generation.
                        txt_lower = text.lower()
                        if status == 401:
                            last_err = f"Auth error ({status})"
                        elif status in (400, 404) and (
                            "not a valid model id" in txt_lower
                            or "no allowed providers" in txt_lower
                        ):
                            last_err = f"Model unavailable: {model_name} ({status})"
                        else:
                            last_err = f"HTTP {status}: {text}"
                        break

                    provider = self._get_provider(model_name)
                    if provider == "gemini":
                        # Gemini returns an empty/absent candidates list on
                        # safety blocks, and candidates without "parts" on
                        # RECITATION/MAX_TOKENS. Treat both as a failed call so
                        # we fall back instead of crashing or returning "".
                        candidates = data.get("candidates") or []
                        if not candidates:
                            block = data.get("promptFeedback") or text[:200]
                            last_err = f"Gemini returned no candidates for {model_name}: {block}"
                            break
                        parts = (candidates[0].get("content") or {}).get("parts") or []
                        content = parts[0].get("text", "") if parts else ""
                        if not content:
                            # finishReason SAFETY/RECITATION/MAX_TOKENS yields a
                            # candidate with no usable text. Treat as a failed
                            # call so we fall back rather than return "".
                            finish = candidates[0].get("finishReason")
                            last_err = f"Gemini returned empty content for {model_name} (finishReason={finish})"
                            break
                        usage = {}  # Gemini API doesn't provide usage stats in the same way
                    else:  # openrouter
                        # OpenRouter sometimes returns HTTP 200 with an
                        # {"error": ...} body and no choices.
                        choices = data.get("choices") or []
                        if not choices:
                            err = data.get("error") or text[:200]
                            last_err = f"No choices returned for {model_name}: {err}"
                            break
                        content = (choices[0].get("message") or {}).get("content")
                        usage = data.get("usage", {})
                    # Coerce a null/absent content to "" so downstream string
                    # operations (strip/len) never hit a None.
                    return GenerationResult(content=content or "", usage=usage, raw_response=data)
                except Exception as e:
                    last_err = str(e)
                    if attempt < self.max_retries:
                        time.sleep(backoff)
                        backoff *= 2
                    else:
                        break
            continue
        if self.enable_mock_fallback:
            mock_text = self._build_mock_response(prompt, self.name)
            return GenerationResult(
                content=mock_text,
                usage={},
                raw_response={"mock": True, "reason": "unavailable_or_auth_failure", "model": self.model, "error": last_err},
                error=last_err,
                is_mock=True,
            )
        return GenerationResult(content="", usage={}, raw_response={}, error=last_err)

class Fusion:
    """Coordinates multi-agent debate and final synthesis."""

    def __init__(
        self,
        api_keys: Dict[str, str],
        agents: List[AgentConfig],
        rounds: int = 3,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        synthesizer: Optional[AgentConfig] = None,
        request_headers: Optional[Dict[str, str]] = None,
        log_file: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        if not agents:
            raise ValueError("Fusion requires at least one agent.")
        names = [cfg.name for cfg in agents]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"Agent names must be unique; debate state is keyed by name. "
                f"Duplicate name(s): {duplicates}"
            )
        self.api_keys = api_keys
        self.rounds = rounds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.request_headers = request_headers or {}
        self.mock_warning_printed = False
        self.agents: List[Agent] = [
            Agent(
                name=cfg.name,
                model=cfg.model,
                api_keys=self.api_keys,
                request_headers=self.request_headers,
                default_system_prompt=self._build_system_prompt(cfg),
                fallback_models=cfg.fallback_models,
                enable_mock_fallback=True,
            )
            for cfg in agents
        ]
        self.synthesizer_cfg = synthesizer if synthesizer is not None else agents[0]
        self.synthesizer = Agent(
            name=f"Synthesizer({self.synthesizer_cfg.name})",
            model=self.synthesizer_cfg.model,
            api_keys=self.api_keys,
            request_headers=self.request_headers,
            default_system_prompt=(
                "You are the synthesizer. Merge inputs into the single best answer, "
                "maximizing clarity, correctness, and completeness."
            ),
            fallback_models=self.synthesizer_cfg.fallback_models,
            enable_mock_fallback=True,
        )
        # Per-instance logger so multiple Fusion objects don't share (and
        # repeatedly clear) a process-global logger's handlers, which leaked
        # file descriptors. Log to stderr so human-readable lines never
        # collide with the progress UI written to stdout.
        self.logger = logging.getLogger(f"fusion.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        for h in list(self.logger.handlers):
            h.close()
            self.logger.removeHandler(h)

        if not log_file:
            os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
            log_file = DEFAULT_RUN_LOG
        sh = logging.StreamHandler(sys.stderr)
        # Only surface warnings/errors on the console; INFO progress is already
        # shown via the progress callback (and the JSONL run log captures the
        # full detail), so emitting it here would just clutter the terminal.
        sh.setLevel(logging.WARNING)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        sh.setFormatter(fmt)
        self.logger.addHandler(sh)
        # The structured JSONL run log is written directly by _log_jsonl; it is
        # deliberately NOT a logging FileHandler target, which would interleave
        # human-readable lines into the JSONL and corrupt it.
        self.run_log_path = log_file

    @staticmethod
    def _build_system_prompt(cfg: AgentConfig) -> str:
        """Create a role-aware system prompt for an agent."""
        role = cfg.role or cfg.name
        return (
            f"You are {cfg.name}, specialized in {role}. "
            "Follow instructions carefully, avoid fabrications, and provide step-by-step, verifiable reasoning when asked."
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Very rough token estimate (~4 chars/token)."""
        return max(1, len(text) // 4)

    def _warn_costs(self, prompts: List[str]) -> None:
        """Emit a warning when a round might be expensive."""
        est_prompt_tokens = sum(self._estimate_tokens(p) for p in prompts)
        est_output_tokens = est_prompt_tokens
        total_est = est_prompt_tokens + est_output_tokens
        if total_est > 6000:
            self.logger.warning(
                f"High token usage estimated: ~{total_est} tokens this round. Consider reducing rounds or agents."
            )

    def _log_jsonl(self, row: Dict[str, Any]) -> None:
        """Append a row to the JSONL run log; ignore failures."""
        try:
            with open(self.run_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write run log: {e}")

    def debate(self, query: str, paper_mode: bool = False, progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Tuple[str, Dict[str, Any]]:
        """Run initial generation, N review rounds, and final synthesis."""
        run_meta: Dict[str, Any] = {
            "query": query,
            "rounds": self.rounds,
            "agents": [a.name for a in self.agents],
            "steps": [],
        }

        if progress_callback:
            progress_callback({"type": "status", "message": "Starting debate..."})

        self.logger.info("Initial generation by all agents...")
        initial_outputs: Dict[str, GenerationResult] = {}
        prompts_for_warn: List[str] = []
        for agent in self.agents:
            if progress_callback:
                progress_callback({"type": "spinner_start", "message": f"Initial • {agent.name}"})
            system = agent.default_system_prompt
            prompt = self._build_initial_prompt(query, paper_mode)
            prompts_for_warn.append(prompt)
            res = agent.generate(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system_prompt=system,
                seed=self.seed,
            )
            initial_outputs[agent.name] = res
            self._log_jsonl({
                "phase": "initial",
                "agent": agent.name,
                "model": agent.model,
                "request": {"prompt": prompt},
                "response": res.raw_response or {"error": res.error, "content": res.content},
            })
            if progress_callback:
                progress_callback({"type": "spinner_stop"})
                if res.is_mock and not self.mock_warning_printed:
                    progress_callback({"type": "status", "message": "Note: Mock response produced due to API access issues.", "color": "red"})
                    self.mock_warning_printed = True
                progress_callback({"type": "endline", "ok": bool(res.content.strip()), "message": f"Initial • {agent.name} ({len(res.content)} chars)"})
            self.logger.info(f"{agent.name} produced initial output ({len(res.content)} chars)")

        self._warn_costs(prompts_for_warn)

        agent_latest: Dict[str, str] = {k: v.content for k, v in initial_outputs.items()}
        # Track which agents produced mock/placeholder output so it can be kept
        # out of other agents' review context and the final synthesis.
        agent_is_mock: Dict[str, bool] = {k: v.is_mock for k, v in initial_outputs.items()}
        for r in range(1, self.rounds + 1):
            round_label = f"Review round {r}/{self.rounds}"
            if progress_callback:
                progress_callback({"type": "status", "message": round_label})
            self.logger.info(f"Review round {r}/{self.rounds}...")
            new_outputs: Dict[str, str] = {}
            new_is_mock: Dict[str, bool] = {}
            prompts_for_warn = []
            for agent in self.agents:
                if progress_callback:
                    progress_callback({"type": "spinner_start", "message": f"Review • {agent.name}"})
                # Don't feed mock/empty contributions to the reviewing agent —
                # asking a real model to critique placeholder text poisons the
                # debate.
                others = {
                    name: content
                    for name, content in agent_latest.items()
                    if name != agent.name and content.strip() and not agent_is_mock.get(name, False)
                }
                prompt = self._build_review_prompt(
                    query=query,
                    self_response=agent_latest.get(agent.name, ""),
                    other_responses=others,
                    paper_mode=paper_mode,
                )
                prompts_for_warn.append(prompt)
                res = agent.generate(
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system_prompt=agent.default_system_prompt,
                    seed=self.seed,
                )
                refined = self._extract_refined(res.content)
                new_outputs[agent.name] = refined
                new_is_mock[agent.name] = res.is_mock
                self._log_jsonl({
                    "phase": f"review_{r}",
                    "agent": agent.name,
                    "model": agent.model,
                    "request": {"prompt": prompt},
                    "response": res.raw_response or {"error": res.error, "content": res.content},
                })
                if progress_callback:
                    progress_callback({"type": "spinner_stop"})
                    if getattr(res, "is_mock", False) and not self.mock_warning_printed:
                        progress_callback({"type": "status", "message": "Note: Mock response produced due to API access issues.", "color": "red"})
                        self.mock_warning_printed = True
                    progress_callback({"type": "endline", "ok": bool(refined.strip()), "message": f"Review • {agent.name} ({len(refined)} chars)"})
                self.logger.info(f"{agent.name} refined output ({len(refined)} chars)")
            agent_latest = new_outputs
            agent_is_mock = new_is_mock
            self._warn_costs(prompts_for_warn)

        if progress_callback:
            progress_callback({"type": "synth_start", "message": "Synthesizing"})
        self.logger.info("Synthesizing final answer...")
        # Synthesize only over genuine, non-empty contributions. If every agent
        # degraded, fall back to whatever we have so the run still produces a
        # (clearly flagged) result rather than synthesizing over nothing.
        synth_inputs = {
            name: content
            for name, content in agent_latest.items()
            if content.strip() and not agent_is_mock.get(name, False)
        }
        if not synth_inputs:
            synth_inputs = agent_latest
        synth_prompt = self._build_synthesis_prompt(query, synth_inputs, paper_mode)
        synth_res = self.synthesizer.generate(
            prompt=synth_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system_prompt=self.synthesizer.default_system_prompt,
            seed=self.seed,
        )
        final_answer = synth_res.content
        self._log_jsonl({
            "phase": "synthesis",
            "agent": self.synthesizer.name,
            "model": self.synthesizer.model,
            "request": {"prompt": synth_prompt},
            "response": synth_res.raw_response or {"error": synth_res.error, "content": synth_res.content},
        })
        if progress_callback:
            progress_callback({"type": "synth_stop"})
            if getattr(synth_res, "is_mock", False) and not self.mock_warning_printed:
                progress_callback({"type": "status", "message": "Note: Mock response produced due to API access issues.", "color": "red"})
                self.mock_warning_printed = True
            progress_callback({"type": "endline", "ok": bool(final_answer.strip()), "message": f"Synthesis ({len(final_answer)} chars)"})
        self.logger.info(f"Synthesis complete ({len(final_answer)} chars)")

        run_meta["final_answer"] = final_answer
        run_meta["final_usage"] = synth_res.usage
        # Machine-readable degradation flags so non-interactive callers (e.g.
        # the benchmark runner) can tell a real synthesis from a mock/empty one.
        run_meta["agent_is_mock"] = agent_is_mock
        run_meta["is_mock"] = synth_res.is_mock
        run_meta["any_mock"] = synth_res.is_mock or any(agent_is_mock.values())
        return final_answer, run_meta

    @staticmethod
    def _build_initial_prompt(query: str, paper_mode: bool) -> str:
        """Prompt template for the initial generation step."""
        if paper_mode:
            return (
                "Paper Writing Mode. Task: Compose a clear, well-structured scholarly response. "
                "Structure your answer with: Abstract, Introduction, Methods/Approach, Results/Findings, Discussion, Conclusion, and References (if applicable).\n\n"
                f"Prompt: {query}"
            )
        return f"Task: Provide the best possible answer to the user's query.\n\nQuery: {query}"

    @staticmethod
    def _build_review_prompt(
        query: str,
        self_response: str,
        other_responses: Dict[str, str],
        paper_mode: bool,
    ) -> str:
        """Prompt template for critical review + refinement."""
        others_str = "\n\n".join([f"[{name}]\n{resp}" for name, resp in other_responses.items()])
        mode_line = "Maintain the requested academic structure. " if paper_mode else ""
        return (
            f"You will review responses from other agents and refine your own. {mode_line}"
            "Instructions:\n"
            "1) Identify factual errors, logical gaps, and unclear explanations in others' responses.\n"
            "2) Suggest concrete improvements and corrections.\n"
            "3) Produce your refined answer that integrates the best ideas and fixes flaws.\n\n"
            f"Original Query:\n{query}\n\n"
            f"Your Previous Answer:\n{self_response}\n\n"
            f"Other Agents' Answers:\n{others_str}\n\n"
            "Output format:\n"
            "- Critique: <your short critique>\n"
            "- Refined Answer: <your improved answer>\n"
        )

    # Match a "Refined Answer:" header only at the start of a line, tolerating
    # leading bullets/markdown bold, so an in-sentence mention inside a critique
    # ("...my refined answer:...") doesn't get mistaken for the header.
    _REFINED_MARKER_RE = re.compile(
        r"^[ \t]*[-*]?[ \t]*\**[ \t]*refined answer[ \t]*\**[ \t]*:[ \t]*\**[ \t]*",
        re.IGNORECASE | re.MULTILINE,
    )

    @staticmethod
    def _extract_refined(text: str) -> str:
        """Heuristic to extract the refined answer block from an agent's output."""
        if not text:
            return ""
        # Mock responses (any variant: "[MOCK]", "[MOCK SYNTHESIS]") don't
        # follow the refined-answer format; pass them through unchanged.
        if text.lstrip().startswith("[MOCK"):
            return text.strip()

        matches = list(Fusion._REFINED_MARKER_RE.finditer(text))
        if matches:
            # Use the last header so a real "Refined Answer:" block wins over an
            # earlier in-critique reference.
            return text[matches[-1].end():].strip()
        return text.strip()

    @staticmethod
    def _build_synthesis_prompt(
        query: str,
        agent_outputs: Dict[str, str],
        paper_mode: bool,
    ) -> str:
        """Prompt template for the final synthesizer call."""
        outputs_joined = "\n\n".join([f"[{name}]\n{content}" for name, content in agent_outputs.items()])
        mode_line = (
            "Ensure academic structure (Abstract, Introduction, Methods, Results, Discussion, Conclusion). "
            if paper_mode
            else ""
        )
        return (
            f"You are the synthesizer. {mode_line}"
            "Merge the following agent answers into a single best response."
            " Be precise, cite assumptions, and avoid contradictions. If there is disagreement, resolve it with reasoning or present consensus with justification.\n\n"
            f"Original Query:\n{query}\n\n"
            f"Agent Answers:\n{outputs_joined}\n\n"
            "Final Answer:"
        )
