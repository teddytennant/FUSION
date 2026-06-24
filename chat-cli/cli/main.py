#!/usr/bin/env python3
"""
FUSION (Federated Unified Systems Integration Orchestration Network) - CLI

This module implements the command-line interface for the FUSION framework.
"""

import argparse
import dataclasses
import json
import os
import sys
import shlex
from typing import Any, Dict, List, Optional, Tuple, Callable
import getpass
import threading
from contextlib import contextmanager
import shutil
import time

# Add framework directory to path to import from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from framework.fusion_core import (
    Fusion,
    AgentConfig,
    Agent,
    DEFAULT_LOG_DIR,
    DEFAULT_RUN_LOG,
    API_CONFIG,
)

class TerminalUI:
    """Terminal UI: ANSI-colored status lines, spinners, and a typewriter
    final-answer renderer. All effects degrade to plain text when stdout is not
    a TTY or when --no-ansi/--no-anim is passed."""

    SPINNER_FRAMES = [
        "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
        "◐", "◑", "◒", "◓",
        "◢", "◣", "◤", "◥",
        "▌", "▐", "▌", "▐",
        "▉", "▊", "▋", "▌", "▍", "▎", "▏", "▎", "▍", "▌", "▋", "▊",
    ]

    # Status icons, matched to message content in print_status/endline.
    TICK = "✓"
    CROSS = "✗"
    WARNING = "!"
    INFO = "i"
    ROCKET = "→"
    BRAIN = "●"
    SPARKLES = "·"
    FIRE = "~"
    GEAR = "○"
    MAGIC = "◇"

    def __init__(self, enable_ansi: bool = True, enable_anim: bool = True, typewriter_ms: int = 0) -> None:
        self.enable_ansi = enable_ansi and sys.stdout.isatty()
        self.enable_anim = enable_anim and sys.stdout.isatty()
        self.typewriter_ms = max(0, int(typewriter_ms))

        # ANSI escapes (empty strings when color is disabled, so callers can
        # interpolate them unconditionally).
        self.COLOR_RESET = "\x1b[0m" if self.enable_ansi else ""
        self.COLOR_DIM = "\x1b[2m" if self.enable_ansi else ""
        self.COLOR_BOLD = "\x1b[1m" if self.enable_ansi else ""

        self.COLOR_CYAN = "\x1b[36m" if self.enable_ansi else ""
        self.COLOR_MAGENTA = "\x1b[35m" if self.enable_ansi else ""
        self.COLOR_BLUE = "\x1b[34m" if self.enable_ansi else ""

        self.COLOR_BRIGHT_CYAN = "\x1b[96m" if self.enable_ansi else ""
        self.COLOR_BRIGHT_GREEN = "\x1b[92m" if self.enable_ansi else ""
        self.COLOR_BRIGHT_RED = "\x1b[91m" if self.enable_ansi else ""
        self.COLOR_BRIGHT_YELLOW = "\x1b[93m" if self.enable_ansi else ""
        self.COLOR_BRIGHT_MAGENTA = "\x1b[95m" if self.enable_ansi else ""
        self.COLOR_BRIGHT_BLUE = "\x1b[94m" if self.enable_ansi else ""
        self.COLOR_BRIGHT_WHITE = "\x1b[97m" if self.enable_ansi else ""

    def _println(self, text: str = "") -> None:
        sys.stdout.write(text + ("\n" if not text.endswith("\n") else ""))
        sys.stdout.flush()

    def print_status(self, text: str, color: Optional[str] = None) -> None:
        """Print a status line, picking an icon/color from keywords in the text."""
        if self.enable_ansi:
            icon = ""
            if "starting" in text.lower():
                icon = f"{self.ROCKET} "
                color = color or self.COLOR_BRIGHT_BLUE
            elif "complete" in text.lower() or "finished" in text.lower():
                icon = f"{self.SPARKLES} "
                color = color or self.COLOR_BRIGHT_GREEN
            elif "error" in text.lower() or "failed" in text.lower():
                icon = f"{self.WARNING} "
                color = color or self.COLOR_BRIGHT_RED
            elif "warning" in text.lower():
                icon = f"{self.WARNING} "
                color = color or self.COLOR_BRIGHT_YELLOW
            elif "info" in text.lower():
                icon = f"{self.INFO} "
                color = color or self.COLOR_BRIGHT_CYAN
            elif "synthesizing" in text.lower():
                icon = f"{self.MAGIC} "
                color = color or self.COLOR_BRIGHT_MAGENTA
            elif "debate" in text.lower():
                icon = f"{self.BRAIN} "
                color = color or self.COLOR_BRIGHT_CYAN
                
            self._println(f"{color}{icon}{text}{self.COLOR_RESET}")
        else:
            self._println(text)

    def print_separator(self, char: str = "═", color: Optional[str] = None) -> None:
        """Print a decorative separator line."""
        width = min(80, self._term_width())
        separator = char * width
        if color and self.enable_ansi:
            self._println(f"{color}{separator}{self.COLOR_RESET}")
        else:
            self._println(separator)

    def _term_width(self) -> int:
        try:
            return max(40, shutil.get_terminal_size((80, 20)).columns)
        except Exception:
            return 80

    def prompt_box(self, title: str = "FUSION", prompt_label: str = "Enter your query") -> str:
        """Draw a bordered input box with a centered title and read one line."""
        width = min(100, self._term_width() - 2)
        inner_w = width - 2
        title_len = len(title) + 2
        title_pad_left = (inner_w - title_len) // 2
        title_pad_right = inner_w - title_len - title_pad_left

        if self.enable_ansi:
            title_text = f" {self.COLOR_BRIGHT_YELLOW}{self.COLOR_BOLD}{title}{self.COLOR_RESET}{self.COLOR_CYAN} "
            top = self.COLOR_CYAN + "╔" + "═" * title_pad_left + title_text + "═" * title_pad_right + "╗" + self.COLOR_RESET
            bottom = self.COLOR_CYAN + "╚" + "═" * inner_w + "╝" + self.COLOR_RESET
            prompt_text = self.COLOR_CYAN + f"║ {self.COLOR_BRIGHT_GREEN}{prompt_label}{self.COLOR_CYAN}: {self.COLOR_BRIGHT_WHITE}"
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
        """Animate a spinner next to `label` on a daemon thread until the
        context exits, then clear the line. No-op when animation is disabled."""
        stop = threading.Event()
        start_time = time.time()

        def run() -> None:
            i = 0
            colors = [self.COLOR_CYAN, self.COLOR_BRIGHT_CYAN, self.COLOR_BLUE, self.COLOR_BRIGHT_BLUE,
                     self.COLOR_MAGENTA, self.COLOR_BRIGHT_MAGENTA]

            while not stop.is_set():
                if self.enable_anim:
                    frame = self.SPINNER_FRAMES[i % len(self.SPINNER_FRAMES)]
                    color = colors[i % len(colors)]
                    elapsed = time.time() - start_time
                    time_str = f" ({elapsed:.1f}s)" if elapsed > 1.0 else ""
                    pulse = " " if i % 2 == 0 else "·"
                    sys.stdout.write(f"\r{color}{frame}{self.COLOR_RESET} {label}{time_str}{pulse}")
                    sys.stdout.flush()
                    i += 1
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
                sys.stdout.write("\r" + " " * 80 + "\r")
                sys.stdout.flush()

    @contextmanager
    def synth_progress(self, label: str = "Synthesizing"):
        """Spinner variant for the synthesis step: a converging/exploding
        particle animation. Same lifecycle as spinner()."""
        stop = threading.Event()
        start_time = time.time()

        fusion_frames = [
            "        ••        ",
            "      »»••««      ",
            "    »»»»»•«««««    ",
            "   »»»»»»•««««««   ",
            "  »»»»»»»•«««««««  ",
            " »»»»»»»»•«««««««« ",
            "»»»»»»»»»•«««««««««",
            "»»»»»»»»💥««««««««",
            "   *   * 💥 *   *   ",
            "  * * *~*💥*~* * *  ",
            "   *   * 💥 *   *   ",
            "        *~*        ",
            "      *~* *~*      ",
            "    *~*  .  *~*    ",
            "  *~*    .    *~*  ",
            " *~*     .     *~* ",
            "*~*      .      *~*",
            " ~*      .      *~ ",
            " *      .      * ",
            "       .       ",
            "      ...      ",
            "               ",
        ]

        def run() -> None:
            i = 0
            while not stop.is_set():
                if self.enable_anim:
                    frame = fusion_frames[i % len(fusion_frames)]
                    if self.enable_ansi:
                        # Colorize only the glyphs the frames actually contain.
                        frame = frame.replace("•", f"{self.COLOR_BRIGHT_YELLOW}•{self.COLOR_RESET}")
                        frame = frame.replace("~", f"{self.COLOR_BRIGHT_RED}~{self.COLOR_RESET}")
                        frame = frame.replace("*", f"{self.COLOR_BRIGHT_MAGENTA}*{self.COLOR_RESET}")
                        frame = frame.replace(".", f"{self.COLOR_DIM}.{self.COLOR_RESET}")
                        frame = frame.replace("»", f"{self.COLOR_CYAN}»{self.COLOR_RESET}")
                        frame = frame.replace("«", f"{self.COLOR_CYAN}«{self.COLOR_RESET}")
                        frame = frame.replace("💥", f"{self.COLOR_BRIGHT_YELLOW}💥{self.COLOR_RESET}")

                    elapsed = time.time() - start_time
                    time_str = f" ({elapsed:.1f}s)"
                    pulse = " ·" if i % 4 == 0 else "  "
                    sys.stdout.write(f"\r{self.COLOR_BRIGHT_BLUE}{label}{self.COLOR_RESET}{time_str} {frame}{pulse}")
                    sys.stdout.flush()
                    i += 1
                time.sleep(0.12)

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
                sys.stdout.write("\r" + " " * 100 + "\r")
                sys.stdout.flush()

    def endline(self, ok: bool, text: str) -> None:
        """Print a completed step line with a check/cross and a keyword icon."""
        if ok:
            icon = self.TICK
            color = self.COLOR_BRIGHT_GREEN
            if "initial" in text.lower():
                icon = f"{self.ROCKET} {self.TICK}"
            elif "review" in text.lower():
                icon = f"{self.BRAIN} {self.TICK}"
            elif "synthesis" in text.lower():
                icon = f"{self.MAGIC} {self.TICK}"
            elif "gemini" in text.lower():
                icon = f"{self.SPARKLES} {self.TICK}"
            elif "grok" in text.lower():
                icon = f"{self.FIRE} {self.TICK}"
            elif "deepseek" in text.lower():
                icon = f"{self.GEAR} {self.TICK}"
        else:
            icon = self.CROSS
            color = self.COLOR_BRIGHT_RED

        if self.enable_ansi:
            self._println(f"{color}{icon}{self.COLOR_RESET} {text}")
        else:
            self._println(f"{icon} {text}")

    def print_ascii_header(self) -> None:
        """Render the FUSION ASCII banner and subtitle."""
        header = r'''
███████╗██╗   ██╗███████╗██╗ ██████╗ ███╗   ██╗
██╔════╝██║   ██║██╔════╝██║██╔═══██╗████╗  ██║
█████╗  ██║   ██║███████╗██║██║   ██║██╔██╗ ██║
██╔══╝  ╚██╗ ██╔╝╚════██║██║██║   ██║██║╚██╗██║
██║      ╚████╔╝ ███████║██║╚██████╔╝██║ ╚████║
╚═╝       ╚═══╝  ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝
        '''
        subtitle = " Federated Unified Systems Integration Orchestration Network"
        tagline = " Multi-Agent AI Debate & Synthesis Engine"
        
        if self.enable_ansi:
            colors = [
                self.COLOR_BRIGHT_MAGENTA, self.COLOR_BRIGHT_CYAN, self.COLOR_BRIGHT_BLUE,
                self.COLOR_BRIGHT_GREEN, self.COLOR_BRIGHT_YELLOW, self.COLOR_BRIGHT_RED
            ]
            lines = header.split('\n')
            for i, line in enumerate(lines):
                if line.strip():
                    color = colors[i % len(colors)]
                    self._println(f"{color}{line}{self.COLOR_RESET}")

            self._println(f"{self.COLOR_BRIGHT_YELLOW}{self.COLOR_BOLD}{subtitle}{self.COLOR_RESET}")
            self._println(f"{self.COLOR_CYAN}{tagline}{self.COLOR_RESET}")

            width = min(80, self._term_width())
            decorative = "═" * width
            self._println(f"{self.COLOR_DIM}{decorative}{self.COLOR_RESET}")
        else:
            self._println(header)
            self._println(subtitle)
            self._println(tagline)

    def typewriter(self, text: str) -> None:
        """Print `text` char-by-char with a blinking cursor when typewriter_ms>0;
        otherwise print it in one shot."""
        if not self.enable_anim or self.typewriter_ms <= 0:
            if self.enable_ansi:
                self._println(f"{self.COLOR_BRIGHT_GREEN}{text}{self.COLOR_RESET}")
            else:
                self._println(text)
            return

        delay = self.typewriter_ms / 1000.0
        cursor_frames = ["▌", "▐", "▌", "▐"]
        cursor_i = 0

        if self.enable_ansi:
            sys.stdout.write(self.COLOR_BRIGHT_GREEN)

        for i, ch in enumerate(text):
            sys.stdout.write(ch)
            if i % 10 == 0:
                cursor = cursor_frames[cursor_i % len(cursor_frames)]
                sys.stdout.write(f"{self.COLOR_BRIGHT_YELLOW}{cursor}{self.COLOR_BRIGHT_GREEN}")
                sys.stdout.write("\b")  # overwrite the cursor on the next char
                cursor_i += 1
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
        "api_keys": {
            provider: os.getenv(config["api_key_env"], "")
            for provider, config in API_CONFIG.items()
        },
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

_AGENT_CONFIG_FIELDS = {f.name for f in dataclasses.fields(AgentConfig)}


def merge_api_keys(base: Dict[str, str], overlay: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Deep-merge api_keys so a partial overlay never wipes existing providers.

    A shallow dict ``update`` replaces the whole ``api_keys`` object, silently
    dropping (for example) an env-loaded OPENROUTER_API_KEY when a config file
    only specifies a gemini key. Only non-empty overlay values override.
    """
    merged = dict(base or {})
    for provider, key in (overlay or {}).items():
        if key:
            merged[provider] = key
    return merged


def agent_config_from_dict(d: Dict[str, Any]) -> AgentConfig:
    """Build an AgentConfig from a dict with a clear error on bad fields."""
    if not isinstance(d, dict):
        raise ValueError(f"Each agent config must be an object, got {type(d).__name__}.")
    unknown = set(d) - _AGENT_CONFIG_FIELDS
    if unknown:
        raise ValueError(
            f"Unknown agent config field(s): {sorted(unknown)}. "
            f"Allowed: {sorted(_AGENT_CONFIG_FIELDS)}."
        )
    missing = {"name", "model"} - set(d)
    if missing:
        raise ValueError(f"Agent config missing required field(s): {sorted(missing)}.")
    return AgentConfig(**d)


def build_fusion_from_config(cfg: Dict[str, Any]) -> Fusion:
    """Merge user config with defaults and build the orchestrator."""
    merged = build_default_config()
    # Deep-merge api_keys first so the shallow update below can't drop providers.
    merged["api_keys"] = merge_api_keys(merged.get("api_keys", {}), cfg.get("api_keys"))
    merged.update({k: v for k, v in cfg.items() if v is not None and k != "api_keys"})

    api_keys: Dict[str, str] = merged.get("api_keys") or {}
    if not any(v and str(v).strip() for v in api_keys.values()):
        raise RuntimeError("Missing API keys. Set at least one via onboarding, env, or config (e.g., OPENROUTER_API_KEY, GEMINI_API_KEY).")

    agent_cfgs = [agent_config_from_dict(a) for a in merged.get("agents", [])]
    synthesizer_cfg = agent_config_from_dict(merged["synthesizer"]) if merged.get("synthesizer") else None  # type: ignore

    fusion = Fusion(
        api_keys=api_keys,
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
            if "initial" in message.lower() or "review" in message.lower():
                ui.print_separator("─", ui.COLOR_DIM)
            spinner_context = ui.spinner(message)
            spinner_context.__enter__()
        elif event_type == "spinner_stop":
            if spinner_context:
                spinner_context.__exit__(None, None, None)
                spinner_context = None
        elif event_type == "synth_start":
            ui.print_separator("═", ui.COLOR_BRIGHT_MAGENTA)
            synth_context = ui.synth_progress(message)
            synth_context.__enter__()
        elif event_type == "synth_stop":
            if synth_context:
                synth_context.__exit__(None, None, None)
                synth_context = None
        elif event_type == "endline":
            ui.endline(event.get("ok", False), message)

    try:
        final_answer, _meta = fusion.debate(query=query, paper_mode=paper_mode, progress_callback=progress_callback)
    finally:
        # Ensure spinner/synth daemon threads are always stopped and the
        # terminal restored, even if debate() raises mid-spinner.
        if spinner_context is not None:
            spinner_context.__exit__(None, None, None)
        if synth_context is not None:
            synth_context.__exit__(None, None, None)
    return final_answer

def run_benchmark(fusion: Fusion, dataset_path: str, output_path: Optional[str]) -> None:
    """Load a simple dataset and evaluate queries sequentially."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Explicit check (not assert: assertions are stripped under `python -O`).
    if not isinstance(data, list):
        raise ValueError("Benchmark dataset must be a JSON list of items.")

    results = []
    degraded = 0
    for i, item in enumerate(data, start=1):
        prompt = item.get("prompt") or item.get("query")
        if not prompt:
            continue
        print(f"[Benchmark] {i}/{len(data)}: {prompt[:80]}...")
        answer, meta = fusion.debate(query=prompt, paper_mode=False)
        is_degraded = bool(meta.get("any_mock"))
        if is_degraded:
            degraded += 1
            print(f"  WARNING: item {item.get('id', i)} produced mock/degraded output "
                  f"(missing/invalid API access) — not a real model answer.")
        results.append({
            "id": item.get("id", i),
            "prompt": prompt,
            "expected": item.get("expected"),
            "answer": answer,
            "degraded": is_degraded,
            "meta": meta,
        })
    if degraded:
        print(f"\nWARNING: {degraded}/{len(results)} benchmark items were mock/degraded. "
              f"Results are not trustworthy until API access is fixed.")
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved benchmark results to {output_path}")
    else:
        print(json.dumps(results[:3], ensure_ascii=False, indent=2))

def prompt_for_api_keys_interactive() -> Dict[str, str]:
    """Prompt user for API keys for all configured providers."""
    print("Welcome to FUSION. To start, you need to provide API keys.")
    print("Paste your keys below (they will not be echoed), or press Enter to skip a provider.")
    api_keys = {}
    for provider, config in API_CONFIG.items():
        prompt_text = f"{config['api_key_env']}: "
        try:
            key = getpass.getpass(prompt=prompt_text)
        except Exception:
            try:
                key = input(prompt_text)
            except EOFError:
                # Non-interactive stdin (CI/pipe): can't onboard, stop asking.
                break
        key = (key or "").strip()
        if key:
            api_keys[provider] = key
    if not any(api_keys.values()):
        print("No keys entered. Aborting onboarding.")
    return api_keys

def maybe_save_env_vars(keys: Dict[str, str], env_path: str = ".env") -> None:
    """Offer to append API keys to a .env file."""
    if not keys:
        return
    choice = input("Save these keys to .env for future runs? [y/N]: ").strip().lower()
    if choice in ("y", "yes"):
        try:
            with open(env_path, "a", encoding="utf-8") as f:
                for provider, key in keys.items():
                    env_var = API_CONFIG[provider]['api_key_env']
                    # shlex.quote prevents a key containing ", $, backticks, or
                    # whitespace from breaking the file or being shell-expanded
                    # when the user later `source`s it.
                    line = f'export {env_var}={shlex.quote(key)}\n'
                    f.write(line)
            print(f"Saved to {env_path}. Next time, run: source {env_path}")
        except Exception as e:
            print(f"Could not save to {env_path}: {e}")

def quick_connectivity_check(keys: Dict[str, str]) -> Tuple[bool, str]:
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
        api_keys=keys,
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
            api_keys=keys,
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

def interactive_onboarding(ui: "TerminalUI") -> Dict[str, str]:
    """Run onboarding: banner, key prompts, connectivity test, save option."""
    ui.print_ascii_header()
    print("This program debates across multiple models and synthesizes a final answer.")
    print("You can paste your API keys now; no terminal exports needed.")
    keys = prompt_for_api_keys_interactive()
    if not keys:
        return {}
    for provider, key in keys.items():
        env_var = API_CONFIG[provider]['api_key_env']
        os.environ[env_var] = key
    ok, _sample = quick_connectivity_check(keys)
    maybe_save_env_vars(keys)
    if ok:
        print("\nReady. You can now ask a question directly.")
    else:
        print("\nWarning: No accessible models were found for the provided key(s). You may need to enable models in your provider's dashboard.")
    return keys

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
    # Deep-merge api_keys so a partial config file can't wipe env-loaded keys.
    cfg["api_keys"] = merge_api_keys(cfg.get("api_keys", {}), file_cfg.get("api_keys"))
    cfg.update({k: v for k, v in file_cfg.items() if v is not None and k != "api_keys"})

    if not args.onboard:
        ui.print_ascii_header()
        if ui.enable_ansi:
            ui.print_status("Welcome to FUSION! Ready to orchestrate multi-agent debates.", ui.COLOR_BRIGHT_CYAN)
            ui.print_status("Tip: Use --paper-mode for structured academic outputs", ui.COLOR_DIM)
            ui.print_separator("─", ui.COLOR_DIM)

    # Gate onboarding on the merged keys (env + config file), not env alone, so
    # a valid config-with-keys run isn't forced into interactive onboarding
    # (which would crash on non-interactive stdin).
    has_usable_key = any(v and str(v).strip() for v in cfg.get("api_keys", {}).values())
    if args.onboard or not has_usable_key:
        keys = interactive_onboarding(ui)
        cfg["api_keys"] = merge_api_keys(cfg.get("api_keys", {}), keys)

    if args.rounds is not None:
        cfg["rounds"] = args.rounds
    if args.temperature is not None:
        cfg["temperature"] = args.temperature
    if getattr(args, "max_tokens") is not None:
        cfg["max_tokens"] = args.max_tokens

    fusion = build_fusion_from_config(cfg)

    if args.log_file:
        # --log-file selects the JSONL run-log target. The human-readable
        # logger stays on stderr (set up in Fusion.__init__) so it never
        # collides with the stdout progress UI.
        fusion.run_log_path = args.log_file

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

    if ui.enable_ansi:
        ui.print_separator("═", ui.COLOR_BRIGHT_GREEN)
        ui.print_status("FUSION Final Answer", ui.COLOR_BRIGHT_GREEN)
        ui.print_separator("═", ui.COLOR_BRIGHT_GREEN)
    else:
        print("\n==== FUSION Final Answer ====\n")

    ui.typewriter(final)

    if ui.enable_ansi:
        ui.print_separator("─", ui.COLOR_DIM)
        ui.print_status("Debate complete! The agents have synthesized their insights.", ui.COLOR_BRIGHT_GREEN)

if __name__ == "__main__":
    main()
