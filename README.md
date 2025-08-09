# FUSION: Federated Unified Systems Integration Orchestration Network

FUSION is a Python multi-agent chatbot framework that tethers multiple LLMs via the OpenRouter API to debate, review, and synthesize responses iteratively. It aims to produce answers that outperform individual models across various fields.

## Features
- Multiple agents (e.g., Gemini 2.5 Pro, Grok-4, DeepSeek) generate and iteratively refine answers
- 2–3 review rounds with critique and refinement prompts
- Final synthesis by a designated synthesizer agent
- OpenRouter integration with retries, backoff, and error handling
- Logging of all intermediate steps to JSONL
- CLI with single query mode and simple benchmark mode
- Optional Paper Writing Mode (structured academic output)
- Interactive ASCII onboarding to paste API key (no shell exports needed)
- Automatic model fallbacks when a preferred model is unavailable for your key
- Terminal UI with spinners and optional typewriter output
- Mock responses when API key is missing/unavailable (clearly warned in UI)

## Requirements
- Python 3.10+
- An OpenRouter API key
- `requests` (optional). If missing, Python's `urllib` is used automatically.
- Optional: PyTorch for advanced features (currently not required)

## Installation
```bash
git clone https://github.com/teddytennant/FUSION
cd FUSION
sudo ln -s "$(pwd)/fusion" /usr/local/bin/fusion
```

## Quick Start
### Onboarding flow (no shell exports)
```bash
fusion --onboard
```
- Paste your `OPENROUTER_API_KEY` when prompted (hidden input).
- Optionally save it to `.env` for future runs.
- A quick connectivity check runs using non‑Llama models (with an optional last‑resort Llama check).
- If you didn’t pass `--query`, you’ll be prompted to type one interactively.

### Non‑interactive (environment or config file)
- Temporary for current shell:
  ```bash
  export OPENROUTER_API_KEY="sk-or-..."
  fusion --query "Explain quantum computing simply." --rounds 2 --temperature 0.6
  ```
- Persist to future shells (zsh):
  ```bash
  echo 'export OPENROUTER_API_KEY="sk-or-..."' >> ~/.zshrc && source ~/.zshrc
  ```
- `.env` file (manual load):
  ```bash
  echo 'export OPENROUTER_API_KEY="sk-or-..."' > .env
  source .env
  ```

## Alternative: Use a config file
Provide the real key value in JSON (no env substitution inside the file):
```json
{
  "api_key": "sk-or-...",
  "agents": [
    {"name": "Gemini 2.5 Pro", "model": "google/gemini-2.5-pro-preview", "role": "general reasoning"},
    {"name": "Grok-4", "model": "x-ai/grok-4", "role": "factual accuracy"},
    {"name": "DeepSeek", "model": "deepseek/deepseek-coder", "role": "coding and math"}
  ],
  "synthesizer": {"name": "Gemini 2.5 Pro", "model": "google/gemini-2.5-pro-preview"},
  "rounds": 3,
  "max_tokens": 1000,
  "temperature": 0.7,
  "headers": {
    "HTTP-Referer": "http://localhost",
    "X-Title": "FUSION"
  }
}
```
Run with:
```bash
fusion --config config.json --query "Explain quantum computing simply."
```

## Usage Reference
- `--onboard`: interactive ASCII onboarding (paste API key, quick connectivity check)
- `--query "..."`: user query; if omitted, you’ll be prompted interactively
- `--rounds N`: number of review rounds (default 3)
- `--temperature X`: sampling temperature
- `--max-tokens N`: per-call max tokens
- `--paper-mode`: enforce academic structure in outputs
- `--config path.json`: JSON config to customize models/headers/etc
- `--log-file path.jsonl`: where to stream JSONL logs (default `logs/runs.jsonl`)
- UI flags:
  - `--no-ansi`: disable colors
  - `--no-anim`: disable spinners/typewriter
  - `--typewriter-ms N`: delay per character in ms for final answer

### Benchmark Mode
Dataset: a JSON array with at least `prompt` per item (optional `id`, `expected`):
```json
[
  {"id": 1, "prompt": "What is overfitting?", "expected": "...optional..."},
  {"id": 2, "prompt": "Explain transformers."}
]
```
Run:
```bash
fusion --benchmark path/to/dataset.json --benchmark-output results.json
```

## Customizing Models
You can add/swap agents via config. Example OpenRouter model IDs:
- `google/gemini-2.5-pro-preview` (primary synthesizer by default)
- `google/gemini-2.0-flash-lite-001`, `google/gemini-flash-1.5-8b` (fallbacks)
- `x-ai/grok-4`, `x-ai/grok-3`, `x-ai/grok-3-mini`
- `deepseek/deepseek-coder`, `deepseek/deepseek-chat`
- Others (Claude, etc.) can be added similarly

### Fallbacks
If a model is unavailable for your key (e.g., 400/404 “not a valid model ID” or “no allowed providers”), FUSION will automatically attempt configured fallbacks per agent and synthesizer before failing.

### Mock Responses
If the API key is missing or a request can’t be served, FUSION returns clearly labeled mock responses so you can still see the debate flow. The UI shows a red warning when mock output is used.

## Notes on Costs
FUSION makes multiple calls per round per agent plus a final synthesis. The app emits approximate token‑usage warnings.

## Troubleshooting
- Prefer `fusion --onboard` to paste your key and run a quick connectivity check.
- If outputs are empty, your key may lack access to the selected models; enable them in OpenRouter or rely on fallbacks.
- The client retries on common rate‑limit/5xx errors with exponential backoff.
- Inspect `logs/runs.jsonl` for intermediate prompts, responses, and metadata.

## Example
```bash
fusion --onboard
```