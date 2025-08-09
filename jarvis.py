#!/usr/bin/env python3
"""
JARVIS (Joint Agents Reviewing Via Iterative Synthesis)

This module implements a multi-agent debate and synthesis framework that
routes all model calls through OpenRouter. Multiple agents generate answers,
critique each other over several rounds, and a synthesizer merges results
into a final response. Includes:
- Robust HTTP handling with retries and simple token estimates
- Optional interactive onboarding (paste API key, quick connectivity check)
- Automatic model fallbacks when preferred models are unavailable
- Basic benchmarking utilities
"""

import argparse
import json
import os
import sys
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import getpass

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
    ) -> None:
        self.name = name
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self.request_headers = request_headers or {}
        self.default_system_prompt = default_system_prompt
        self.fallback_models = list(fallback_models or [])
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
        # If we reach here, all models failed
        return GenerationResult(content="", usage={}, raw_response={}, error=last_err)


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

        # Instantiate debate agents
        self.agents: List[Agent] = [
            Agent(
                name=cfg.name,
                model=cfg.model,
                api_key=self.api_key,
                request_headers=self.request_headers,
                default_system_prompt=self._build_system_prompt(cfg),
                fallback_models=cfg.fallback_models,
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

    def debate(self, query: str, paper_mode: bool = False) -> Tuple[str, Dict[str, Any]]:
        """Run initial generation, N review rounds, and final synthesis."""
        run_meta: Dict[str, Any] = {
            "query": query,
            "rounds": self.rounds,
            "agents": [a.name for a in self.agents],
            "steps": [],
        }

        # 1) Initial answers from all agents
        self.logger.info("Initial generation by all agents...")
        initial_outputs: Dict[str, GenerationResult] = {}
        prompts_for_warn: List[str] = []
        for agent in self.agents:
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
            self.logger.info(f"{agent.name} produced initial output ({len(res.content)} chars)")

        self._warn_costs(prompts_for_warn)

        # 2) Iterative reviews/refinements
        agent_latest: Dict[str, str] = {k: v.content for k, v in initial_outputs.items()}
        for r in range(1, self.rounds + 1):
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
                self.logger.info(f"{agent.name} refined output ({len(refined)} chars)")
            agent_latest = new_outputs
            self._warn_costs(prompts_for_warn)

        # 3) Final synthesis
        self.logger.info("Synthesizing final answer...")
        synth_prompt = self._build_synthesis_prompt(query, agent_latest, paper_mode)
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


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load a JSON config from disk, or return empty dict if None."""
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_default_config() -> Dict[str, Any]:
    """Default configuration with reasonable models and fallbacks."""
    return {
        "api_key": os.getenv("OPENROUTER_API_KEY", ""),
        "agents": [
            {
                "name": "Gemini 2.5 Pro",
                "model": "google/gemini-2.5-pro-preview",
                "role": "general reasoning",
                "fallback_models": [
                    "google/gemini-2.0-flash-lite-001",
                    "google/gemini-flash-1.5-8b",
                ],
            },
            {
                "name": "Grok-4",
                "model": "x-ai/grok-4",
                "role": "factual accuracy",
                "fallback_models": [
                    "x-ai/grok-3",
                    "x-ai/grok-3-mini",
                ],
            },
            {
                "name": "DeepSeek",
                "model": "deepseek/deepseek-coder",
                "role": "coding and math",
                "fallback_models": [
                    "deepseek/deepseek-chat",
                ],
            },
        ],
        "synthesizer": {
            "name": "Gemini 2.5 Pro",
            "model": "google/gemini-2.5-pro-preview",
            "fallback_models": [
                "google/gemini-2.0-flash-lite-001",
                "google/gemini-flash-1.5-8b",
            ],
        },
        "rounds": 3,
        "max_tokens": 1000,
        "temperature": 0.7,
        "headers": {},
        "seed": None,
    }


def build_jarvis_from_config(cfg: Dict[str, Any]) -> Jarvis:
    """Merge user config with defaults and build the orchestrator."""
    merged = build_default_config()
    merged.update({k: v for k, v in cfg.items() if v is not None})

    api_key: str = merged.get("api_key") or ""
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY. Set it via onboarding, env, or config.")

    agent_cfgs = [AgentConfig(**a) for a in merged.get("agents", [])]
    synthesizer_cfg = AgentConfig(**merged["synthesizer"]) if merged.get("synthesizer") else None  # type: ignore

    jarvis = Jarvis(
        api_key=api_key,
        agents=agent_cfgs,
        rounds=int(merged.get("rounds", 3)),
        max_tokens=int(merged.get("max_tokens", 1000)),
        temperature=float(merged.get("temperature", 0.7)),
        synthesizer=synthesizer_cfg,
        request_headers=merged.get("headers", {}),
        seed=merged.get("seed"),
    )
    return jarvis


def run_single_query(jarvis: Jarvis, query: str, paper_mode: bool = False) -> str:
    """Convenience wrapper for single interactive runs."""
    final_answer, _meta = jarvis.debate(query=query, paper_mode=paper_mode)
    return final_answer


def run_benchmark(jarvis: Jarvis, dataset_path: str, output_path: Optional[str]) -> None:
    """Load a simple dataset and evaluate queries sequentially."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "Benchmark dataset must be a list of items."

    results = []
    for i, item in enumerate(data, start=1):
        prompt = item.get("prompt") or item.get("query")
        if not prompt:
            continue
        print(f"[Benchmark] {i}/{len(data)}: {prompt[:80]}...")
        answer, meta = jarvis.debate(query=prompt, paper_mode=False)
        results.append({
            "id": item.get("id", i),
            "prompt": prompt,
            "expected": item.get("expected"),
            "answer": answer,
            "meta": meta,
        })
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved benchmark results to {output_path}")
    else:
        print(json.dumps(results[:3], ensure_ascii=False, indent=2))  # preview


def print_ascii_header() -> None:
    """Render the JARVIS ASCII banner."""
    header = r"""
      ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
      ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
      ██║███████║██████╔╝██║   ██║██║███████╗
 ██   ██║██╔══██║██╔══██╗██║   ██║██║╚════██║
 ╚█████╔╝██║  ██║██║  ██║╚██████╔╝██║███████║
  ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝╚══════╝

 Joint Agents Reviewing Via Iterative Synthesis
    """
    print(header)


def prompt_for_api_key_interactive() -> Optional[str]:
    """Prompt the user for an OpenRouter API key (hidden input)."""
    print("Welcome to JARVIS. To start, you need an OpenRouter API key.")
    print("Paste your key below (it will not be echoed), or press Enter to cancel.")
    try:
        key = getpass.getpass(prompt="OPENROUTER_API_KEY: ")
    except Exception:
        key = input("OPENROUTER_API_KEY: ")
    key = (key or "").strip()
    if not key:
        print("No key entered. Aborting onboarding.")
        return None
    return key


def maybe_save_env_var(key: str, env_path: str = ".env") -> None:
    """Offer to append the API key to a .env file for convenience."""
    choice = input("Save this key to .env for future runs? [y/N]: ").strip().lower()
    if choice in ("y", "yes"):
        try:
            line = f"export OPENROUTER_API_KEY=\"{key}\"\n"
            with open(env_path, "a", encoding="utf-8") as f:
                f.write(line)
            print(f"Saved to {env_path}. Next time, run: source {env_path}")
        except Exception as e:  # noqa: BLE001
            print(f"Could not save to {env_path}: {e}")


def quick_connectivity_check(key: str) -> Tuple[bool, str]:
    """Fire a short request across several non‑Llama models to verify access.

    Optionally offers a consented last-resort Llama test.
    """
    candidate_models = [
        "google/gemini-2.0-flash-lite-001",
        "google/gemini-flash-1.5-8b",
        "x-ai/grok-3-mini",
        "x-ai/grok-3",
    ]
    print("\nRunning a quick connectivity check...\n")
    test_agent = Agent(
        name="ConnectivityCheck",
        model=candidate_models[0],
        api_key=key,
        max_retries=1,
        timeout=30,
        request_headers={},
        default_system_prompt="You are a minimal assistant.",
        fallback_models=candidate_models[1:],
    )
    res = test_agent.generate(prompt="Reply with OK.", max_tokens=16, temperature=0.0)
    if res.content.strip():
        print("Connectivity check succeeded.")
        return True, res.content.strip()
    consent = input("Non-Llama models failed. Try Llama 3.1 8B as a last resort? [y/N]: ").strip().lower()
    if consent in ("y", "yes"):
        llama_agent = Agent(
            name="ConnectivityCheckLlama",
            model="meta-llama/llama-3.1-8b-instruct",
            api_key=key,
            max_retries=1,
            timeout=30,
            request_headers={},
            default_system_prompt="You are a minimal assistant.",
        )
        res2 = llama_agent.generate(prompt="Reply with OK.", max_tokens=16, temperature=0.0)
        if res2.content.strip():
            print("Connectivity check succeeded with Llama.")
            return True, res2.content.strip()
    print("Connectivity check failed.")
    return False, ""


def interactive_onboarding() -> Optional[str]:
    """Run the ASCII onboarding: banner, key prompt, connectivity test, save option."""
    print_ascii_header()
    print("This program debates across multiple models via OpenRouter and synthesizes a final answer.")
    print("You can paste your API key now; no terminal exports needed.")
    key = prompt_for_api_key_interactive()
    if not key:
        return None
    os.environ["OPENROUTER_API_KEY"] = key
    ok, _sample = quick_connectivity_check(key)
    maybe_save_env_var(key)
    if ok:
        print("\nReady. You can now ask a question directly.")
        return key
    print("\nWarning: No accessible models were found for this key. You may need to enable models in OpenRouter.")
    return key


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Define CLI arguments for interactive and scripted usage."""
    parser = argparse.ArgumentParser(
        description="JARVIS: Joint Agents Reviewing Via Iterative Synthesis",
    )
    parser.add_argument(
        "--onboard",
        action="store_true",
        help="Run interactive onboarding (paste API key, quick connectivity check)",
    )
    parser.add_argument("--query", type=str, help="User query to answer")
    parser.add_argument("--rounds", type=int, default=None, help="Number of debate/review rounds (overrides config)")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (overrides config)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens for each generation (overrides config)")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file")
    parser.add_argument("--paper-mode", action="store_true", help="Enable paper writing mode (structured academic output)")
    parser.add_argument("--benchmark", type=str, default=None, help="Path to a benchmark JSON dataset (list of {id,prompt,expected})")
    parser.add_argument("--benchmark-output", type=str, default=None, help="Where to write benchmark results (JSON)")
    parser.add_argument("--log-file", type=str, default=None, help="Optional log file path (JSONL). Defaults to logs/runs.jsonl")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry: parse args, onboard if needed, build Jarvis, run query/benchmark."""
    args = parse_args(argv)
    file_cfg = load_config(args.config)
    cfg = build_default_config()
    cfg.update({k: v for k, v in file_cfg.items() if v is not None})

    # Onboarding when requested or when no API key present
    if args.onboard or not (cfg.get("api_key") or os.getenv("OPENROUTER_API_KEY")):
        interactive_onboarding()
        cfg["api_key"] = os.getenv("OPENROUTER_API_KEY", "")

    # CLI overrides for quick experimentation
    if args.rounds is not None:
        cfg["rounds"] = args.rounds
    if args.temperature is not None:
        cfg["temperature"] = args.temperature
    if getattr(args, "max_tokens") is not None:
        cfg["max_tokens"] = args.max_tokens

    # Build orchestrator
    jarvis = build_jarvis_from_config(cfg)

    # Optional log file override
    if args.log_file:
        for h in list(jarvis.logger.handlers):
            jarvis.logger.removeHandler(h)
        fh = logging.FileHandler(args.log_file)
        sh = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        jarvis.logger.addHandler(fh)
        jarvis.logger.addHandler(sh)
        jarvis.run_log_path = args.log_file

    # Benchmark mode exits after run
    if args.benchmark:
        run_benchmark(jarvis, args.benchmark, args.benchmark_output)
        return

    # Interactive prompt when no --query provided
    if not args.query:
        print("\nEnter your query (or press Enter to exit):")
        try:
            q = input("> ").strip()
        except EOFError:
            q = ""
        if not q:
            print("No query provided. Exiting.")
            return
        args.query = q

    # Execute
    final = run_single_query(jarvis, args.query, paper_mode=args.paper_mode)
    print("\n==== JARVIS Final Answer ====\n")
    print(final)


if __name__ == "__main__":
    main() 