# FUSION Project Structure

FUSION has been reorganized into three main components for better modularity and separation of concerns.

## Directory Structure

```
FUSION/
├── framework/           # Core multi-agent framework
│   ├── fusion_core.py   # Main orchestration logic
│   └── logs/           # Runtime logs
├── chat-cli/           # Multi-agent debate CLI
│   └── cli/
│       └── main.py     # Chat CLI entry point
├── code-cli/           # Coding assistant CLI (based on Groq Code CLI)
│   ├── src/           # TypeScript source code
│   ├── dist/          # Compiled JavaScript
│   ├── package.json   # Node.js dependencies
│   └── README.md      # Code CLI documentation
├── fusion              # Chat CLI launcher script
├── fusion-code         # Code CLI launcher script
└── README.md          # Main project documentation
```

## Components

### 1. Framework (`framework/`)
- **Purpose**: Core multi-agent orchestration logic
- **Language**: Python
- **Key Files**: 
  - `fusion_core.py` - Main Fusion class and Agent implementations
  - `logs/` - Runtime logs and debugging information

### 2. Chat CLI (`chat-cli/`)
- **Purpose**: Multi-agent debate and discussion interface
- **Language**: Python
- **Usage**: `fusion --query "Your question"`
- **Features**:
  - Multi-agent debates with 2-3 review rounds
  - Final synthesis by designated agent
  - Paper writing mode
  - Benchmark mode
  - Interactive onboarding

### 3. Code CLI (`code-cli/`)
- **Purpose**: Coding assistant with file operations
- **Language**: TypeScript/Node.js
- **Usage**: `fusion-code`
- **Features**:
  - File reading, writing, and editing
  - Command execution
  - Code analysis and generation
  - Interactive chat interface
  - Tool-based operations

## Installation

```bash
# Clone the repository
git clone https://github.com/teddytennant/FUSION
cd FUSION

# Install chat CLI
sudo ln -s "$(pwd)/fusion" /usr/local/bin/fusion

# Install code CLI
cd code-cli
npm install
npm run build
npm link
cd ..
```

## Usage

### Chat CLI (Multi-Agent Debate)
```bash
# Interactive mode
fusion

# Direct query
fusion --query "Explain quantum computing"

# Onboarding
fusion --onboard
```

### Code CLI (Coding Assistant)
```bash
# Start coding session
fusion-code

# Use /login to set up API key
# Use /help to see available commands
# Use tools to read, create, and edit files
```

## API Keys

Both CLIs use OpenRouter API keys:
- **Environment Variable**: `OPENROUTER_API_KEY`
- **Get Key**: https://openrouter.ai/keys
- **Supported Models**: Gemini 2.5 Pro, Grok-4, DeepSeek, and others

## Development

### Chat CLI Development
- Edit files in `chat-cli/` and `framework/`
- Test with `fusion --query "test"`

### Code CLI Development
- Edit TypeScript files in `code-cli/src/`
- Run `npm run dev` for watch mode
- Test with `fusion-code`

## Key Differences

| Feature | Chat CLI | Code CLI |
|---------|----------|----------|
| **Purpose** | Multi-agent debates | Coding assistance |
| **Language** | Python | TypeScript |
| **Interface** | Command-line arguments | Interactive TUI |
| **Tools** | Debate orchestration | File operations |
| **Output** | Synthesized answers | Code files |
| **Best For** | Complex questions | Development tasks | 