#!/usr/bin/env python3
"""
JARVIS (Joint Agents Reviewing Via Iterative Synthesis) - CLI

This module implements the command-line interface for the JARVIS framework.
"""

import argparse
import json
import os
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple
import getpass

# Add framework directory to path to import from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from framework.jarvis_core import (
    Jarvis,
    AgentConfig,
    TerminalUI,
    Agent,
    DEFAULT_LOG_DIR,
    DEFAULT_RUN_LOG,
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


def run_single_query(jarvis: Jarvis, query: str, paper_mode: bool = False, ui: Optional[TerminalUI] = None) -> str:
    """Convenience wrapper for single interactive runs."""
    final_answer, _meta = jarvis.debate(query=query, paper_mode=paper_mode, ui=ui)
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
            line = f'export OPENROUTER_API_KEY="{key}"\n'
            with open(env_path, "a", encoding="utf-8") as f:
                f.write(line)
            print(f"Saved to {env_path}. Next time, run: source {env_path}")
        except Exception as e:  # noqa: BLE001
            print(f"Could not save to {env_path}: {e}")


def quick_connectivity_check(key: str) -> Tuple[bool, str]:
    """Fire a short request across several non\u2011Llama models to verify access.

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


def interactive_onboarding(ui: "TerminalUI") -> Optional[str]:
    """Run the ASCII onboarding: banner, key prompt, connectivity test, save option."""
    ui.print_ascii_header()
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
    # UI flags
    parser.add_argument("--no-ansi", action="store_true", help="Disable ANSI colors and styling")
    parser.add_argument("--no-anim", action="store_true", help="Disable spinner and typewriter animations")
    parser.add_argument("--typewriter-ms", type=int, default=0, help="Final answer typewriter delay per char in ms (0 to disable)")
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
    ui = TerminalUI(enable_ansi=(not args.no_ansi), enable_anim=(not args.no_anim), typewriter_ms=args.typewriter_ms)

    file_cfg = load_config(args.config)
    cfg = build_default_config()
    cfg.update({k: v for k, v in file_cfg.items() if v is not None})

    # Show banner when not running onboarding explicitly
    if not args.onboard:
        ui.print_ascii_header()

    # Onboarding when requested or when no API key present
    if args.onboard or not (cfg.get("api_key") or os.getenv("OPENROUTER_API_KEY")):
        interactive_onboarding(ui)
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
        sh = logging.StreamHandler(sys.stderr if ui.enable_anim else sys.stdout)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        jarvis.logger.addHandler(fh)
        jarvis.logger.addHandler(sh)
        jarvis.run_log_path = args.log_file
    else:
        # Re-route console logs to stderr when animating to avoid spinner clashes
        for h in jarvis.logger.handlers:
            if isinstance(h, logging.StreamHandler):
                h.stream = sys.stderr if ui.enable_anim else sys.stdout

    # Benchmark mode exits after run
    if args.benchmark:
        run_benchmark(jarvis, args.benchmark, args.benchmark_output)
        return

    # Interactive prompt when no --query provided
    if not args.query:
        # Fancy boxed prompt when animations are enabled; fallback to plain input otherwise
        if ui.enable_anim:
            q = ui.prompt_box(title="JARVIS", prompt_label="Enter your query")
        else:
            print("\nEnter your query (or press Enter to exit):")
            try:
                q = input("> ").strip()
            except EOFError:
                q = ""
        if not q:
            print("No query provided. Exiting.")
            return
        args.query = q

    final = run_single_query(jarvis, args.query, paper_mode=args.paper_mode, ui=ui)
    print("\n==== JARVIS Final Answer ====\n")
    # Typewriter if enabled
    ui.typewriter(final)


if __name__ == "__main__":
    main()
