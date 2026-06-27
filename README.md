# FUSION

A multi-agent LLM debate CLI. Several models answer the same query, critique each
other over N review rounds, then a synthesizer merges them into one answer — all
routed through [OpenRouter](https://openrouter.ai).

It's an open-source take on "model fusion" — running several frontier models
against the same prompt and combining their answers instead of trusting any single
one. Because every call goes through OpenRouter, you pick the exact roster: mix
models from different labs, swap them per run, and pay per token with no lock-in.
In practice the fusion of independent models tends to beat the best single model in
it, since they catch each other's mistakes during the review rounds.

## What roster to run

From experience, the strongest pairing has been **Claude Opus 4.8 + GLM 5.2**. Two
strong, independent models debating each other generally land at or above the
quality of a three-model roster — adding a third model has diminishing returns and
costs more latency and tokens. If you do want a third voice, **GPT 5.5** is the one
I'd add.

So: start with two models (Opus 4.8 + GLM 5.2). Add GPT 5.5 only if you have a
reason to.

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

# Recommended two-model fusion: Opus 4.8 + GLM 5.2.
[[agents]]
model = "anthropic/claude-opus-4.8"
role = "analyst"
fallback_models = ["anthropic/claude-sonnet-4.6"]

[[agents]]
model = "z-ai/glm-5.2"
fallback_models = ["z-ai/glm-4.6"]

# Add a third agent only if you want it — GPT 5.5 is the one to reach for:
# [[agents]]
# model = "openai/gpt-5.5"
# fallback_models = ["openai/gpt-5.1"]

[synthesizer]
model = "anthropic/claude-opus-4.8"
fallback_models = ["z-ai/glm-5.2"]
```

- `agents` — each has a `model`, optional `role`, and an ordered list of
  `fallback_models`.
- `synthesizer` — the agent that merges the final answers (also has fallbacks).
- `rounds`, `max_tokens`, `temperature` — debate-wide defaults.

Model ids are OpenRouter routes, so any provider OpenRouter exposes works — mix and
match across labs (Anthropic, OpenAI, Z.ai, Google, xAI, DeepSeek, and others).

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

## On the "model fusion" trend

Fusing several frontier models into one answer is suddenly fashionable. OpenRouter
shipped its own multi-agent debate product literally called **Fusion**
([openrouter.ai/fusion](https://openrouter.ai/fusion)), and Sakura AI's **fugu** is
landing around the same time. Both are arriving now, in 2026.

This FUSION has been around since **August 2025** — well before either. OpenRouter
even reused the name, which is a funny thing to discover about a project you've
already been running for the better part of a year. The idea isn't novel and nobody
owns it, but for the record: the name and the approach here predate the commercial
versions.

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
