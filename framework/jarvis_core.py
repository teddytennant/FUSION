#!/usr/bin/env python3
"""
JARVIS (Joint Agents Reviewing Via Iterative Synthesis) - Core Framework

This module contains the core classes for the JARVIS framework, including:
- Agent: A wrapper for language model APIs.
- Jarvis: The orchestrator for multi-agent debates.
- TerminalUI: A lightweight terminal UI toolkit.
"""

import json
import os
import sys
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import getpass
import threading
from contextlib import contextmanager
import shutil

# Optional torch import for advanced features (e.g., cosine similarity)
try:
    import torch  # noqa: F401
except Exception:
    torch = None  # type: ignore

# HTTP client: prefer requests if available, else fallback to urllib
try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore
    import urllib.request
    import urllib.error


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
DEFAULT_RUN_LOG = os.path.join(DEFAULT_LOG_DIR, "runs.jsonl")


@dataclass
class AgentConfig:
    """Declarative configuration for an agent.

    name: Human-readable label for the agent
    model: OpenRouter model id (e.g., "google/gemini-2.5-pro-preview")
    role: Optional specialization label for prompts
    fallback_models: Optional ordered list of fallback model ids to try if the
                     primary model is unavailable for the current API key
    """
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
    """Thin wrapper around OpenRouter's Chat Completions for a single model.

    Adds:
    - system prompt support
    - retries + exponential backoff
    - optional persistent HTTP session reuse
    - ordered fallback model attempts for availability issues
    """

    def __init__(
        self,
        name: str,
        model: str,
        api_key: str,
        max_retries: int = 3,
        timeout: int = 60,
        request_headers: Optional[Dict[str, str]] = None,
        default_system_prompt: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        enable_mock_fallback: bool = True,
    ) -> None:
        self.name = name
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self.request_headers = request_headers or {}
        self.default_system_prompt = default_system_prompt
        self.fallback_models = list(fallback_models or [])
        self.enable_mock_fallback = enable_mock_fallback
        # Persist a session when requests exists to reuse TCP connections
        self.session = None
        if requests is not None:
            try:
                self.session = requests.Session()
            except Exception:
                self.session = None

    def _post(self, headers: Dict[str, str], payload: Dict[str, Any]) -> Tuple[int, str, Dict[str, Any]]:
        """Send a POST request via requests (preferred) or urllib fallback.

        Returns (status_code, text_body, parsed_json_if_any)
        """
        if requests is not None:
            http = self.session or requests
            resp = http.post(
                url=OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            status = resp.status_code
            text = resp.text
            data = resp.json() if status < 400 else {}
            return status, text, data
        # urllib fallback path
        req = urllib.request.Request(
            OPENROUTER_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:  # type: ignore
            data = json.loads(r.read().decode("utf-8"))
            status = 200
            text = ""
        return status, text, data

    @staticmethod
    def _build_mock_response(prompt: str, agent_name: str) -> str:
        summary = prompt.strip().split("\n")[0]
        summary = (summary[:140] + "…") if len(summary) > 140 else summary
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
        # Build messages array, adding optional system + extra messages
        messages: List[Dict[str, str]] = []
        effective_system_prompt = system_prompt or self.default_system_prompt
        if effective_system_prompt:
            messages.append({"role": "system", "content": effective_system_prompt})
        if extra_messages:
            messages.extend(extra_messages)
        messages.append({"role": "user", "content": prompt})

        # Prepare shared headers once for all attempts
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.request_headers,
        }

        # If API key is missing and mock fallback is allowed, return mock
        if self.enable_mock_fallback and not (self.api_key and self.api_key.strip()):
            mock_text = self._build_mock_response(prompt, self.name)
            return GenerationResult(
                content=mock_text,
                usage={},
                raw_response={"mock": True, "reason": "missing_api_key", "model": self.model},
                error=None,
                is_mock=True,
            )

        # Attempt primary model first, then fallbacks
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
                    status, text, data = self._post(headers=headers, payload=payload)
                    if status >= 400:
                        # Switch models on 400/404 with availability messages
                        txt_lower = text.lower()
                        if (
                            status in (400, 404)
                            and (
                                "not a valid model id" in txt_lower
                                or "no allowed providers" in txt_lower
                            )
                        ):
                            last_err = f"Model unavailable: {model_name} ({status})"
                            break  # advance to next model
                        # Retry transient conditions
                        if status in (408, 409, 429) or 500 <= status < 600:
                            raise RuntimeError(f"HTTP {status}: {text}")
                        # Auth or other permanent error
                        if status in (401,):
                            last_err = f"Auth error ({status})"
                            break
                        # Non-retry error: return as-is
                        return GenerationResult(
                            content="",
                            usage={},
                            raw_response={"status": status, "text": text},
                            error=f"HTTP {status}: {text}",
                        )
                    # Success path
                    content = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    usage = data.get("usage", {})
                    return GenerationResult(content=content, usage=usage, raw_response=data)
                except Exception as e:  # noqa: BLE001
                    # Exponential backoff across retryable exceptions
                    last_err = str(e)
                    if attempt < self.max_retries:
                        time.sleep(backoff)
                        backoff *= 2
                    else:
                        break
            # Next fallback model
            continue
        # If we reach here, all models failed -> return mock if allowed
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


class TerminalUI:
    """Lightweight terminal UI with spinners, colored status, and typewriter output.

    Avoids external deps. Uses ANSI codes only if enabled and stdout is a TTY.
    """

    SPINNER_FRAMES = [
        "⠋", "⠙", "⠚", "⠞", "⠖", "⠦", "⠴", "⠲", "⠳", "⠓"
    ]
    TICK = "✔"
    CROSS = "✘"

    def __init__(self, enable_ansi: bool = True, enable_anim: bool = True, typewriter_ms: int = 0) -> None:
        self.enable_ansi = enable_ansi and sys.stdout.isatty()
        self.enable_anim = enable_anim and sys.stdout.isatty()
        self.typewriter_ms = max(0, int(typewriter_ms))
        # Colors
        self.COLOR_RESET = "\x1b[0m" if self.enable_ansi else ""
        self.COLOR_DIM = "\x1b[2m" if self.enable_ansi else ""
        self.COLOR_CYAN = "\x1b[36m" if self.enable_ansi else ""
        self.COLOR_GREEN = "\x1b[32m" if self.enable_ansi else ""
        self.COLOR_RED = "\x1b[31m" if self.enable_ansi else ""
        self.COLOR_YELLOW = "\x1b[33m" if self.enable_ansi else ""
        self.COLOR_MAGENTA = "\x1b[35m" if self.enable_ansi else ""
        self.COLOR_BLUE = "\x1b[34m" if self.enable_ansi else ""

    def _println(self, text: str = "") -> None:
        sys.stdout.write(text + ("\n" if not text.endswith("\n") else ""))
        sys.stdout.flush()

    def print_status(self, text: str, color: Optional[str] = None) -> None:
        if color and self.enable_ansi:
            self._println(f"{color}{text}{self.COLOR_RESET}")
        else:
            self._println(text)

    def _term_width(self) -> int:
        try:
            return max(40, shutil.get_terminal_size((80, 20)).columns)
        except Exception:
            return 80

    def prompt_box(self, title: str = "JARVIS", prompt_label: str = "Enter your query") -> str:
        """Draw a more stylish box for input."""
        width = min(100, self._term_width() - 2)
        inner_w = width - 2
        
        title_len = len(title) + 2
        title_pad_left = (inner_w - title_len) // 2
        title_pad_right = inner_w - title_len - title_pad_left

        if self.enable_ansi:
            title_text = f" {self.COLOR_YELLOW}{title}{self.COLOR_CYAN} "
            top = self.COLOR_CYAN + "╔" + "═" * title_pad_left + title_text + "═" * title_pad_right + "╗" + self.COLOR_RESET
            bottom = self.COLOR_CYAN + "╚" + "═" * inner_w + "╝" + self.COLOR_RESET
            prompt_text = self.COLOR_CYAN + f"║ {prompt_label}: " + self.COLOR_RESET
        else:
            top = "┌" + "─" * title_pad_left + f" {title} " + "─" * title_pad_right + "┐"
            bottom = "└" + "─" * inner_w + "┘"
            prompt_text = f"│ {prompt_label}: "

        self._println(top)
        try:
            user_input = input(prompt_text)
        except EOFError:
            user_input = ""
        self._println(bottom)
        return user_input

    @contextmanager
    def spinner(self, label: str):
        """Context manager spinner: shows label with animated glyph until block exits."""
        stop = threading.Event()
        line_lock = threading.Lock()

        def run() -> None:
            i = 0
            while not stop.is_set():
                if self.enable_anim:
                    frame = TerminalUI.SPINNER_FRAMES[i % len(TerminalUI.SPINNER_FRAMES)]
                    sys.stdout.write(f"\r{self.COLOR_CYAN}{frame}{self.COLOR_RESET} {label}")
                    sys.stdout.flush()
                    i += 1
                time.sleep(0.1)

        t: Optional[threading.Thread] = None
        if self.enable_anim:
            t = threading.Thread(target=run, daemon=True)
            t.start()
        try:
            yield
        finally:
            stop.set()
            if t is not None:
                t.join(timeout=0.2)
            # Clear spinner line
            if self.enable_anim:
                sys.stdout.write("\r" + " " * (len(label) + 4) + "\r")
                sys.stdout.flush()

    @contextmanager
    def synth_progress(self, label: str = "Synthesizing", bar_width: int = 30):
        """A more dynamic progress animation."""
        stop = threading.Event()
        width = max(10, min(bar_width, self._term_width() - len(label) - 10))
        
        def run() -> None:
            pos = 0
            direction = 1
            scanner_width = 5
            while not stop.is_set():
                if self.enable_anim:
                    bar = ["─"] * width
                    start = pos
                    end = min(width, pos + scanner_width)
                    for j in range(start, end):
                        bar[j] = "━"

                    bar_line = "".join(bar)
                    if self.enable_ansi:
                        sys.stdout.write(f"\r{self.COLOR_BLUE}{label}{self.COLOR_RESET} [{self.COLOR_MAGENTA}{bar_line}{self.COLOR_RESET}]")
                    else:
                        sys.stdout.write(f"\r{label} [{bar_line}]")
                    sys.stdout.flush()
                    
                    pos += direction
                    if pos >= width - scanner_width:
                        direction = -1
                        pos = width - scanner_width
                    if pos <= 0:
                        direction = 1
                        pos = 0
                time.sleep(0.08)
        
        t: Optional[threading.Thread] = None
        if self.enable_anim:
            t = threading.Thread(target=run, daemon=True)
            t.start()
        try:
            yield
        finally:
            stop.set()
            if t is not None:
                t.join(timeout=0.2)
            if self.enable_anim:
                clear_len = len(label) + width + 6
                sys.stdout.write("\r" + " " * clear_len + "\r")
                sys.stdout.flush()

    def endline(self, ok: bool, text: str) -> None:
        icon = self.TICK if ok else self.CROSS
        color = self.COLOR_GREEN if ok else self.COLOR_RED
        if self.enable_ansi:
            self._println(f"{color}{icon}{self.COLOR_RESET} {text}")
        else:
            self._println(f"{icon} {text}")

    def print_ascii_header(self) -> None:
        """Render the JARVIS ASCII banner with colors."""
        header = r"""
      ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
      ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
      ██║███████║██████╔╝██║   ██║██║███████╗
 ██   ██║██╔══██║██╔══██╗██║   ██║██║╚════██║
 ╚█████╔╝██║  ██║██║  ██║╚██████╔╝██║███████║
  ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝╚══════╝
        """
        subtitle = " Joint Agents Reviewing Via Iterative Synthesis"
        if self.enable_ansi:
            colors = [self.COLOR_MAGENTA, self.COLOR_CYAN, self.COLOR_BLUE]
            lines = header.split('\n')
            for i, line in enumerate(lines):
                if line.strip():
                    self._println(colors[i % len(colors)] + line)
            self._println(self.COLOR_YELLOW + subtitle)
        else:
            self._println(header)
            self._println(subtitle)

    def typewriter(self, text: str) -> None:
        if not self.enable_anim or self.typewriter_ms <= 0:
            self._println(self.COLOR_GREEN + text + self.COLOR_RESET if self.enable_ansi else text)
            return
        delay = self.typewriter_ms / 1000.0
        if self.enable_ansi:
            sys.stdout.write(self.COLOR_GREEN)
        for ch in text:
            sys.stdout.write(ch)
            sys.stdout.flush()
            time.sleep(delay)
        if self.enable_ansi:
            sys.stdout.write(self.COLOR_RESET)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
            sys.stdout.flush()


class Jarvis:
    """Coordinates multi-agent debate and final synthesis."""

    def __init__(
        self,
        api_key: str,
        agents: List[AgentConfig],
        rounds: int = 3,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        synthesizer: Optional[AgentConfig] = None,
        request_headers: Optional[Dict[str, str]] = None,
        log_file: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        # Core settings
        self.api_key = api_key
        self.rounds = rounds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.request_headers = request_headers or {}
        self.mock_warning_printed = False

        # Instantiate debate agents
        self.agents: List[Agent] = [
            Agent(
                name=cfg.name,
                model=cfg.model,
                api_key=self.api_key,
                request_headers=self.request_headers,
                default_system_prompt=self._build_system_prompt(cfg),
                fallback_models=cfg.fallback_models,
                enable_mock_fallback=True,
            )
            for cfg in agents
        ]
        # Synthesizer defaults to first agent if not provided
        self.synthesizer_cfg = synthesizer if synthesizer is not None else agents[0]
        self.synthesizer = Agent(
            name=f"Synthesizer({self.synthesizer_cfg.name})",
            model=self.synthesizer_cfg.model,
            api_key=self.api_key,
            request_headers=self.request_headers,
            default_system_prompt=(
                "You are the synthesizer. Merge inputs into the single best answer, "
                "maximizing clarity, correctness, and completeness."
            ),
            fallback_models=self.synthesizer_cfg.fallback_models,
            enable_mock_fallback=True,
        )

        # Structured logging to both stdout and a JSONL file
        self.logger = logging.getLogger("jarvis")
        self.logger.setLevel(logging.INFO)
        if not log_file:
            os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
            log_file = DEFAULT_RUN_LOG
        fh = logging.FileHandler(log_file)
        sh = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)
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
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Failed to write run log: {e}")

    def debate(self, query: str, paper_mode: bool = False, ui: Optional[TerminalUI] = None) -> Tuple[str, Dict[str, Any]]:
        """Run initial generation, N review rounds, and final synthesis."""
        run_meta: Dict[str, Any] = {
            "query": query,
            "rounds": self.rounds,
            "agents": [a.name for a in self.agents],
            "steps": [],
        }

        if ui:
            ui.print_status("Starting debate...", ui.COLOR_YELLOW)

        # 1) Initial answers from all agents
        self.logger.info("Initial generation by all agents...")
        initial_outputs: Dict[str, GenerationResult] = {}
        prompts_for_warn: List[str] = []
        for agent in self.agents:
            system = agent.default_system_prompt
            prompt = self._build_initial_prompt(query, paper_mode)
            prompts_for_warn.append(prompt)
            label = f"Initial • {agent.name}"
            if ui:
                with ui.spinner(label):
                    res = agent.generate(
                        prompt=prompt,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        system_prompt=system,
                        seed=self.seed,
                    )
            else:
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
            if ui:
                if res.is_mock and not self.mock_warning_printed:
                    ui.print_status("Note: Mock response produced due to API access issues.", ui.COLOR_RED)
                    self.mock_warning_printed = True
                ui.endline(bool(res.content.strip()), f"Initial • {agent.name} ({len(res.content)} chars)")
            self.logger.info(f"{agent.name} produced initial output ({len(res.content)} chars)")

        self._warn_costs(prompts_for_warn)

        # 2) Iterative reviews/refinements
        agent_latest: Dict[str, str] = {k: v.content for k, v in initial_outputs.items()}
        for r in range(1, self.rounds + 1):
            round_label = f"Review round {r}/{self.rounds}"
            if ui:
                ui.print_status(round_label, ui.COLOR_YELLOW)
            self.logger.info(f"Review round {r}/{self.rounds}...")
            new_outputs: Dict[str, str] = {}
            prompts_for_warn = []
            for agent in self.agents:
                others = {name: content for name, content in agent_latest.items() if name != agent.name}
                prompt = self._build_review_prompt(
                    query=query,
                    self_response=agent_latest.get(agent.name, ""),
                    other_responses=others,
                    paper_mode=paper_mode,
                )
                prompts_for_warn.append(prompt)
                label = f"Review • {agent.name}"
                if ui:
                    with ui.spinner(label):
                        res = agent.generate(
                            prompt=prompt,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            system_prompt=agent.default_system_prompt,
                            seed=self.seed,
                        )
                else:
                    res = agent.generate(
                        prompt=prompt,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        system_prompt=agent.default_system_prompt,
                        seed=self.seed,
                    )
                refined = self._extract_refined(res.content)
                new_outputs[agent.name] = refined
                self._log_jsonl({
                    "phase": f"review_{r}",
                    "agent": agent.name,
                    "model": agent.model,
                    "request": {"prompt": prompt},
                    "response": res.raw_response or {"error": res.error, "content": res.content},
                })
                if ui:
                    if getattr(res, "is_mock", False) and not self.mock_warning_printed:
                        ui.print_status("Note: Mock response produced due to API access issues.", ui.COLOR_RED)
                        self.mock_warning_printed = True
                    ui.endline(bool(refined.strip()), f"Review • {agent.name} ({len(refined)} chars)")
                self.logger.info(f"{agent.name} refined output ({len(refined)} chars)")
            agent_latest = new_outputs
            self._warn_costs(prompts_for_warn)

        # 3) Final synthesis
        if ui:
            ui.print_status("Synthesizing final answer...", ui.COLOR_YELLOW)
        self.logger.info("Synthesizing final answer...")
        synth_prompt = self._build_synthesis_prompt(query, agent_latest, paper_mode)
        label = "Synthesis"
        if ui:
            with ui.synth_progress(label):
                synth_res = self.synthesizer.generate(
                    prompt=synth_prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system_prompt=self.synthesizer.default_system_prompt,
                    seed=self.seed,
                )
        else:
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
        if ui:
            if getattr(synth_res, "is_mock", False) and not self.mock_warning_printed:
                ui.print_status("Note: Mock response produced due to API access issues.", ui.COLOR_RED)
                self.mock_warning_printed = True
            ui.endline(bool(final_answer.strip()), f"Synthesis ({len(final_answer)} chars)")
        self.logger.info(f"Synthesis complete ({len(final_answer)} chars)")

        run_meta["final_answer"] = final_answer
        run_meta["final_usage"] = synth_res.usage
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

    @staticmethod
    def _extract_refined(text: str) -> str:
        """Heuristic to extract the refined answer block from an agent's output."""
        lower = text.lower()
        marker = "refined answer:"
        if marker in lower:
            idx = lower.index(marker) + len(marker)
            return text[idx:].strip()
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
