# FUSION

A multi-agent LLM debate CLI. Several models answer the same query, critique each
other over a few review rounds, and a synthesizer merges the results into one
answer. Every call goes through [OpenRouter](https://openrouter.ai), so the roster
is any mix of models it serves.

## What roster to run

I run **Claude Opus 4.8 + GLM 5.2**, and that's the default roster. A third model
adds latency and tokens for not much gain. If you want one anyway, I'd add
**GPT 5.5**.

## How a debate runs

1. **Initial generation:** every agent answers the query on its own.
2. **Review rounds:** for `rounds` iterations, each agent reads the others' latest
   answers, critiques them and writes a refined answer.
3. **Synthesis:** the synthesizer merges the final answers into one response.

Each agent retries a model up to 3 times with backoff on 408, 409, 429, 5xx or
transport errors. Any other failure, or running out of retries, moves it to the next
of its `fallback_models`.

## Install

Prebuilt binaries for Linux and macOS (x86_64 and aarch64):

```bash
curl -fsSL https://raw.githubusercontent.com/teddytennant/FUSION/master/install.sh | sh
```

The script installs to `/usr/local/bin` if it's writable, otherwise `~/.local/bin`.
Set `FUSION_INSTALL_DIR` to pick another directory, or `FUSION_VERSION=v0.1.0` to pin
a release. You can also download `fusion-linux-x86_64`, `fusion-linux-aarch64`,
`fusion-macos-x86_64` or `fusion-macos-aarch64` from the
[releases page](https://github.com/teddytennant/FUSION/releases/latest) and
`chmod +x` it.

The current release, v0.1.0, predates `--serve`. To use server mode, build from
source with a stable Rust toolchain:

```bash
cargo build --release   # binary at target/release/fusion
```

## Setup

FUSION needs an OpenRouter API key. Either run the wizard, which checks the key
with a cheap completion, asks for a roster and round count, and writes
`~/.config/fusion/config.toml`:

```bash
fusion --onboard
```

or export the key:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

## Usage

```bash
fusion --query "Explain quicksort"
fusion            # no query: interactive REPL, `exit` or Ctrl-D to quit
fusion --query "Compare two database isolation levels" --paper-mode --rounds 1
fusion --query "..." --config ./my-config.toml --no-progress
```

| Flag            | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| `-q, --query`   | The query to debate. Omit for the REPL.                      |
| `--onboard`     | Run the setup wizard and exit.                               |
| `--config`      | Config file path (overrides the default location).           |
| `--paper-mode`  | Ask for an academic paper structure in the prompts.          |
| `--rounds`      | Number of review rounds.                                     |
| `--temperature` | Sampling temperature.                                        |
| `--max-tokens`  | Max tokens per completion.                                   |
| `--log-file`    | Where to write the JSONL run log.                            |
| `--no-progress` | Turn off the progress spinner.                               |
| `--serve`       | Serve an OpenAI-compatible API instead of running a query.   |
| `--host`        | Bind address for `--serve` (default `127.0.0.1`).            |
| `--port`        | Port for `--serve` (default `8080`).                         |

The answer goes to stdout and progress to stderr, so piping works.

## Server mode

`fusion --serve` puts the debate behind the OpenAI Chat Completions API, so
anything that takes a custom OpenAI base URL (Codex CLI, opencode, the SDKs) can
use a whole debate as one model.

```bash
fusion --serve
fusion --serve --host 0.0.0.0 --port 11434

export OPENAI_BASE_URL="http://127.0.0.1:8080/v1"
export OPENAI_API_KEY="unused"   # clients want one; FUSION ignores it
```

- `POST /v1/chat/completions` runs a debate over the request's `messages` and
  returns the synthesized answer. `stream: true` works, but the answer arrives as
  one chunk followed by `[DONE]`.
- `GET /v1/models` lists one model, `fusion`.
- `GET /health` returns `ok`.

The request's `model` is echoed back but doesn't pick anything; the roster always
comes from your config. There's no auth on the server, so keep it on localhost or
behind your own proxy. Claude Code speaks the Anthropic Messages API, so it needs a
translating proxy in front.

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"fusion","messages":[{"role":"user","content":"Explain quicksort"}]}'
```

## Config

TOML at `~/.config/fusion/config.toml` on Linux (the platform config dir
elsewhere). Without a file, FUSION uses the built-in Opus 4.8 + GLM 5.2 roster.

```toml
rounds = 2
max_tokens = 4096
temperature = 0.7

[[agents]]
name = "Claude Opus 4.8"
model = "anthropic/claude-opus-4.8"
role = "general reasoning"
fallback_models = ["anthropic/claude-sonnet-4.6"]

[[agents]]
name = "GLM 5.2"
model = "z-ai/glm-5.2"
fallback_models = ["z-ai/glm-4.6"]

# Optional third agent:
# [[agents]]
# name = "GPT 5.5"
# model = "openai/gpt-5.5"
# fallback_models = ["openai/gpt-5.1"]

[synthesizer]
name = "Synthesizer"
model = "anthropic/claude-opus-4.8"
fallback_models = ["z-ai/glm-5.2"]
```

Every agent and the synthesizer need a unique `name` and a `model` (an OpenRouter
model id). `role` and `fallback_models` are optional. Other top-level keys:
`api_key`, `seed`, `log_file`, and `extra_headers` (a table of HTTP headers sent with
every OpenRouter request).

Precedence, low to high: built-in defaults, config file, `OPENROUTER_API_KEY`, CLI
flags.

Each debate appends one JSONL row per step (phase, agent, model, prompt, response)
to `~/.local/share/fusion/runs.jsonl` on Linux, or to `--log-file`.

## Background

FUSION started in August 2025 as a Python proof of concept and was rewritten in
Rust in June 2026 with the same debate algorithm. I haven't benchmarked the debate
against a single model. OpenRouter has since shipped its own
multi-model product under the same name,
[openrouter.ai/fusion](https://openrouter.ai/fusion).

## Development

```bash
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt
```

CI runs fmt, clippy, tests and a release build on every push and pull request.

## License

MIT, see [LICENSE](LICENSE).
