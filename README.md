# FUSION

A multi-agent LLM debate CLI. Several models answer the same query, critique each
other over N review rounds, then a synthesizer merges them into one answer — all
routed through [OpenRouter](https://openrouter.ai).

## How it works

A debate runs in three phases:

1. **Initial generation** — every configured agent answers the query independently.
2. **Review rounds** — for `rounds` iterations, each agent sees the others' latest
   answers, critiques them, and emits a refined answer.
3. **Synthesis** — a synthesizer agent merges the final answers into a single
   response.

Each agent wraps OpenRouter chat-completion calls with retries and per-agent
fallback models, so a failing model degrades gracefully to the next one.

## Install / build

Requires a stable Rust toolchain.

```bash
cargo build --release
```

The binary is produced at `target/release/fusion`.

## Setup

FUSION needs an OpenRouter API key. There are two ways to provide it:

- **Interactive wizard** — run `fusion --onboard`. It validates the key against
  OpenRouter and writes it to `~/.config/fusion/config.toml`.
- **Environment variable** — set `OPENROUTER_API_KEY` in your shell.

```bash
fusion --onboard
# or
export OPENROUTER_API_KEY="sk-or-..."
```

## Usage

```bash
# Single query
fusion --query "Explain quicksort"

# Interactive mode (no query → REPL)
fusion

# Academic structure, one review round
fusion --query "Compare two database isolation levels" --paper-mode --rounds 1

# Use a specific config file, no progress UI
fusion --query "..." --config ./my-config.toml --no-progress
```

### Flags

| Flag            | Description                                                   |
| --------------- | ------------------------------------------------------------- |
| `--query`       | The query to debate. Omit for an interactive REPL.            |
| `--onboard`     | Run the interactive key-onboarding wizard and exit.           |
| `--config`      | Path to a config file (overrides the default location).       |
| `--paper-mode`  | Force an academic structure in the prompts.                   |
| `--rounds`      | Number of review rounds (overrides config).                   |
| `--temperature` | Sampling temperature (overrides config).                      |
| `--max-tokens`  | Max tokens per completion (overrides config).                 |
| `--log-file`    | Path to write the JSONL run log (overrides the default).      |
| `--no-progress` | Disable the progress/spinner UI.                              |

## Config

Configuration lives in a TOML file at `~/.config/fusion/config.toml` on Linux
(the platform config dir elsewhere). The schema, briefly:

```toml
rounds = 2
max_tokens = 4096
temperature = 0.7

[[agents]]
model = "google/gemini-2.5-pro"
role = "analyst"
fallback_models = ["google/gemini-2.5-flash"]

[[agents]]
model = "x-ai/grok-3"
fallback_models = ["x-ai/grok-3-mini"]

[synthesizer]
model = "deepseek/deepseek-chat"
fallback_models = ["deepseek/deepseek-r1"]
```

- `agents` — each has a `model`, optional `role`, and an ordered list of
  `fallback_models`.
- `synthesizer` — the agent that merges the final answers (also has fallbacks).
- `rounds`, `max_tokens`, `temperature` — debate-wide defaults.

Model ids are OpenRouter routes, so any provider OpenRouter exposes works
(Gemini, Grok, DeepSeek, and others).

**Precedence** (low → high): built-in defaults < config file <
`OPENROUTER_API_KEY` env var < CLI flags.

## Run logs

Every debate appends JSONL rows — one per step, recording the phase, agent, model,
request, and response — to a run log. By default this lives under the platform
data dir (`~/.local/share/fusion/runs.jsonl` on Linux). Override the path with
`--log-file`.

## About this rewrite

FUSION is a Rust rewrite of an earlier Python proof-of-concept. The debate
algorithm — initial generation, N review rounds, synthesis — is preserved. The
"debate improves answers" premise was never rigorously benchmarked in the original,
and that hasn't changed here: treat the architecture as the artifact, not any
particular result.

## Development

```bash
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt
```

CI runs these checks (formatting, clippy with warnings denied, tests, and a
release build) on every push and pull request.

## License

MIT — see [LICENSE](LICENSE).
