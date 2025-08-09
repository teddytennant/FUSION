#!/usr/bin/env python3
"""
FUSION (Federated Unified Systems Integration Orchestration Network) - CLI

This module implements the command-line interface for the FUSION framework.
"""

import argparse
import json
import os
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
import getpass
import threading
from contextlib import contextmanager
import shutil
import time

# Add framework directory to path to import from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from framework.fusion_core import (
    Fusion,
    AgentConfig,
    Agent,
    DEFAULT_LOG_DIR,
    DEFAULT_RUN_LOG,
)

class TerminalUI:
    """Lightweight terminal UI with spinners, colored status, and typewriter output."""

    SPINNER_FRAMES = [
        " ( o       + ) ",
        " (  o     +  ) ",
        " (   o   +   ) ",
        " (    o +    ) ",
        " (     o+    ) ",
        " (     *     ) ",
        " (    *.*    ) ",
        " (   * . *   ) ",
        " (  *  .  *  ) ",
        " ( *   .   * ) ",
        " (           ) ",
    ]
    TICK = "✔"
    CROSS = "✘"

    def __init__(self, enable_ansi: bool = True, enable_anim: bool = True, typewriter_ms: int = 0) -> None:
        self.enable_ansi = enable_ansi and sys.stdout.isatty()
        self.enable_anim = enable_anim and sys.stdout.isatty()
        self.typewriter_ms = max(0, int(typewriter_ms))
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

    def prompt_box(self, title: str = "FUSION", prompt_label: str = "Enter your query") -> str:
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
        def run() -> None:
            i = 0
            while not stop.is_set():
                if self.enable_anim:
                    frame = self.SPINNER_FRAMES[i % len(self.SPINNER_FRAMES)]
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
            if self.enable_anim:
                sys.stdout.write("\r" + " " * (len(label) + len(self.SPINNER_FRAMES[0]) + 2) + "\r")
                sys.stdout.flush()

    @contextmanager
    def synth_progress(self, label: str = "Synthesizing"):
        """A fusion-themed progress animation."""
        stop = threading.Event()
        fusion_frames = [
            "  <-- o | | o -->  ",
            "   <--o | | o-->   ",
            "    <--o| |o-->    ",
            "     <--oo-->     ",
            "      <---->      ",
            "     -======-     ",
            "   *--======--*   ",
            "  * *--====--* *  ",
            " * . *--==--* . * ",
            "* . . *----* . . *",
            " . . . *  * . . . ",
            "  . . .    . . .  ",
            "   . .      . .   ",
            "    .        .    ",
            "                    ",
        ]

        def run() -> None:
            i = 0
            while not stop.is_set():
                if self.enable_anim:
                    frame = fusion_frames[i % len(fusion_frames)]
                    if self.enable_ansi:
                        frame = frame.replace("o", f"{self.COLOR_YELLOW}o{self.COLOR_CYAN}")
                        frame = frame.replace("=", f"{self.COLOR_RED}={self.COLOR_CYAN}")
                        frame = frame.replace("*", f"{self.COLOR_MAGENTA}*{self.COLOR_CYAN}")
                        frame = f"{self.COLOR_CYAN}{frame}{self.COLOR_RESET}"

                    sys.stdout.write(f"\r{self.COLOR_BLUE}{label}{self.COLOR_RESET} {frame}")
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
            if self.enable_anim:
                width = len(label) + 2 + len(fusion_frames[0])
                sys.stdout.write("\r" + " " * width + "\r")
                sys.stdout.flush()

    def endline(self, ok: bool, text: str) -> None:
        icon = self.TICK if ok else self.CROSS
        color = self.COLOR_GREEN if ok else self.COLOR_RED
        if self.enable_ansi:
            self._println(f"{color}{icon}{self.COLOR_RESET} {text}")
        else:
            self._println(f"{icon} {text}")

    def print_ascii_header(self) -> None:
        """Render the FUSION ASCII banner with colors."""
        header = r'''
███████╗██╗   ██╗███████╗██╗ ██████╗ ███╗   ██╗
██╔════╝██║   ██║██╔════╝██║██╔═══██╗████╗  ██║
█████╗  ██║   ██║███████╗██║██║   ██║██╔██╗ ██║
██╔══╝  ╚██╗ ██╔╝╚════██║██║██║   ██║██║╚██╗██║
██║      ╚████╔╝ ███████║██║╚██████╔╝██║ ╚████║
╚═╝       ╚═══╝  ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝
        '''
        subtitle = " Federated Unified Systems Integration Orchestration Network"
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

def build_fusion_from_config(cfg: Dict[str, Any]) -> Fusion:
    """Merge user config with defaults and build the orchestrator."""
    merged = build_default_config()
    merged.update({k: v for k, v in cfg.items() if v is not None})

    api_key: str = merged.get("api_key") or ""
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY. Set it via onboarding, env, or config.")

    agent_cfgs = [AgentConfig(**a) for a in merged.get("agents", [])]
    synthesizer_cfg = AgentConfig(**merged["synthesizer"]) if merged.get("synthesizer") else None  # type: ignore

    fusion = Fusion(
        api_key=api_key,
        agents=agent_cfgs,
        rounds=int(merged.get("rounds", 3)),
        max_tokens=int(merged.get("max_tokens", 1000)),
        temperature=float(merged.get("temperature", 0.7)),
        synthesizer=synthesizer_cfg,
        request_headers=merged.get("headers", {}),
        seed=merged.get("seed"),
    )
    return fusion

def run_single_query(fusion: Fusion, query: str, paper_mode: bool = False, ui: Optional[TerminalUI] = None) -> str:
    """Convenience wrapper for single interactive runs."""
    
    spinner_context = None
    synth_context = None

    def progress_callback(event: Dict[str, Any]):
        nonlocal spinner_context, synth_context
        if not ui:
            return
        
        event_type = event.get("type")
        message = event.get("message", "")
        
        if event_type == "status":
            ui.print_status(message, color=event.get("color"))
        elif event_type == "spinner_start":
            spinner_context = ui.spinner(message)
            spinner_context.__enter__()
        elif event_type == "spinner_stop":
            if spinner_context:
                spinner_context.__exit__(None, None, None)
                spinner_context = None
        elif event_type == "synth_start":
            synth_context = ui.synth_progress(message)
            synth_context.__enter__()
        elif event_type == "synth_stop":
            if synth_context:
                synth_context.__exit__(None, None, None)
                synth_context = None
        elif event_type == "endline":
            ui.endline(event.get("ok", False), message)

    final_answer, _meta = fusion.debate(query=query, paper_mode=paper_mode, progress_callback=progress_callback)
    return final_answer

def run_benchmark(fusion: Fusion, dataset_path: str, output_path: Optional[str]) -> None:
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
        answer, meta = fusion.debate(query=prompt, paper_mode=False)
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
        print(json.dumps(results[:3], ensure_ascii=False, indent=2))

def prompt_for_api_key_interactive() -> Optional[str]:
    """Prompt the user for an OpenRouter API key (hidden input)."""
    print("Welcome to FUSION. To start, you need an OpenRouter API key.")
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
        except Exception as e:
            print(f"Could not save to {env_path}: {e}")

def quick_connectivity_check(key: str) -> Tuple[bool, str]:
    """Fire a short request across several non‑Llama models to verify access."""
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
        description="FUSION: Federated Unified Systems Integration Orchestration Network",
    )
    parser.add_argument(
        "--onboard",
        action="store_true",
        help="Run interactive onboarding (paste API key, quick connectivity check)",
    )
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
    """CLI entry: parse args, onboard if needed, build Fusion, run query/benchmark."""
    args = parse_args(argv)
    ui = TerminalUI(enable_ansi=(not args.no_ansi), enable_anim=(not args.no_anim), typewriter_ms=args.typewriter_ms)

    file_cfg = load_config(args.config)
    cfg = build_default_config()
    cfg.update({k: v for k, v in file_cfg.items() if v is not None})

    if not args.onboard:
        ui.print_ascii_header()

    if args.onboard or not (cfg.get("api_key") or os.getenv("OPENROUTER_API_KEY")):
        interactive_onboarding(ui)
        cfg["api_key"] = os.getenv("OPENROUTER_API_KEY", "")

    if args.rounds is not None:
        cfg["rounds"] = args.rounds
    if args.temperature is not None:
        cfg["temperature"] = args.temperature
    if getattr(args, "max_tokens") is not None:
        cfg["max_tokens"] = args.max_tokens

    fusion = build_fusion_from_config(cfg)

    if args.log_file:
        for h in list(fusion.logger.handlers):
            fusion.logger.removeHandler(h)
        fh = logging.FileHandler(args.log_file)
        sh = logging.StreamHandler(sys.stderr if ui.enable_anim else sys.stdout)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        fusion.logger.addHandler(fh)
        fusion.logger.addHandler(sh)
        fusion.run_log_path = args.log_file
    else:
        for h in fusion.logger.handlers:
            if isinstance(h, logging.StreamHandler):
                h.stream = sys.stderr if ui.enable_anim else sys.stdout

    if args.benchmark:
        run_benchmark(fusion, args.benchmark, args.benchmark_output)
        return

    if not args.query:
        if ui.enable_anim:
            q = ui.prompt_box(title="FUSION", prompt_label="Enter your query")
        else:
            print("\nEnter your query (or press Enter to exit):\n")
            try:
                q = input("> ").strip()
            except EOFError:
                q = ""
        if not q:
            print("No query provided. Exiting.")
            return
        args.query = q

    final = run_single_query(fusion, args.query, paper_mode=args.paper_mode, ui=ui)
    print("\n==== FUSION Final Answer ====\n")
    ui.typewriter(final)

if __name__ == "__main__":
    main()
