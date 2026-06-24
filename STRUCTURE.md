# FUSION Project Structure

## Directory Layout

```
FUSION/
├── framework/
│   ├── fusion_core.py   # Agent + Fusion classes: provider routing, debate loop, synthesis
│   └── logs/            # runs.jsonl — per-step JSONL run log
├── chat-cli/
│   └── cli/
│       └── main.py      # CLI: arg parsing, onboarding, config, terminal UI
├── tests/               # unittest suite for the framework and CLI config layer
├── fusion               # launcher script (symlink-safe wrapper around main.py)
├── README.md
└── STRUCTURE.md
```

## Components

### Framework (`framework/`)
Core multi-agent orchestration, pure Python (no required third-party deps;
`requests` is used if present, otherwise `urllib`).

- `Agent` — wraps a chat-completion call. Routes slash-prefixed model IDs
  (e.g. `google/gemini-2.5-pro-preview`) to OpenRouter and bare `gemini-*`
  names to the native Gemini API. Handles retries with exponential backoff,
  per-agent fallback models, and a labeled mock response when no key is set.
- `Fusion` — runs the debate: initial generation from every agent, N review
  rounds where each agent critiques the others and emits a refined answer,
  then a final synthesis. Every step is appended to `logs/runs.jsonl`.

### Chat CLI (`chat-cli/`)
Terminal front end over the framework: interactive key onboarding, JSON config
for agents/models/rounds, paper-writing mode, a benchmark runner over a JSON
dataset, and the spinner/typewriter UI.

## Installation

```bash
git clone https://github.com/teddytennant/FUSION
cd FUSION
# optional: put it on PATH
sudo ln -s "$(pwd)/fusion" /usr/local/bin/fusion
```

## Usage

```bash
./fusion                                   # interactive
./fusion --query "Explain quantum computing"
./fusion --query "..." --rounds 2 --paper-mode
./fusion --onboard                         # paste API keys, run a connectivity check
```

## API Keys

Read from the environment (or entered via `--onboard`):

- `OPENROUTER_API_KEY` — https://openrouter.ai/keys
- `GEMINI_API_KEY` — optional, enables the native Gemini route for bare
  `gemini-*` model names

## Development

- Edit `framework/fusion_core.py` and `chat-cli/cli/main.py`.
- Run the tests: `python -m unittest discover -s tests`.
- Logs land in `framework/logs/runs.jsonl`.
