# FUSION

A multi-agent LLM debate framework I built in August 2025, before multi-agent orchestration setups were a commonplace pattern. Multiple models answer the same query, critique each other over review rounds, and a synthesizer merges the results into one answer.

## Update (June 2026)

OpenRouter has officially shipped multi-agent debate/orchestration as a built-in feature — and they even copied the name: [openrouter.ai/fusion](https://openrouter.ai/fusion). I'm not sure whether it was effectively a direct copy of this framework, but the resemblance is hard to ignore: it carries the exact same "Fusion" name, it runs the same core pipeline (multiple models generate, critique each other over review rounds, then a synthesizer merges them into one answer), it's built on OpenRouter's own multi-model routing just like this project was, and it ships well after FUSION was public in August 2025. I have not officially heard back from them.

## Status

Preserved as-is, as proof of work — not maintained. If I needed this today I'd reach for something like [Karpathy's llm-council](https://github.com/karpathy/llm-council) or another modern multi-agent framework rather than keep building this one. My project [axon](https://github.com/teddytennant/axon) is effectively the better, decentralized evolution of the same idea.

## What it actually does

Pure Python, no framework dependencies (`requests` optional, falls back to `urllib`):

- **`framework/fusion_core.py`** — the core. An `Agent` class wraps chat-completion calls with provider routing (OpenRouter for slash-prefixed model IDs, native Gemini API for bare `gemini-*` names), retries with exponential backoff, per-agent fallback models, and clearly-labeled mock responses when no API key is available. A `Fusion` class runs the debate loop: initial generation from every agent, N review rounds where each agent critiques the others and emits a refined answer, then a final synthesis call. Every step is appended to a JSONL run log, with rough token-cost warnings.
- **`chat-cli/`** — terminal CLI on top of the framework: interactive key onboarding, JSON config for agents/models/rounds, a paper-writing mode (forces academic structure in the prompts), a simple benchmark mode over a JSON dataset, and spinner/typewriter terminal UI.
- **`tests/`** — pytest unit tests for the Agent and Fusion classes.

There was also a `code-cli` (a coding-assistant fork pulled in as a submodule); it broke and was removed, along with its `fusion-code` launcher.

The "debate improves answers" premise was never rigorously benchmarked — the benchmark mode just runs prompts through the pipeline and saves outputs. Treat the architecture, not the results, as the artifact.

## Running it

```bash
git clone https://github.com/teddytennant/FUSION
cd FUSION
./fusion --onboard          # paste an OpenRouter (and optionally Gemini) key
./fusion --query "Explain quantum computing simply." --rounds 2
```

Agents, models, fallbacks, and the synthesizer are configurable via `--config config.json`; see `chat-cli/cli/main.py` for the flags. Logs land in `framework/logs/runs.jsonl`.

## License

MIT
