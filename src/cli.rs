//! Command-line interface: argument parsing and top-level dispatch.
//!
//! NOTE: Stage-0 stub pinning the public API. The real implementation (config
//! merge with flags, onboarding dispatch, REPL/one-shot query, wiring the
//! provider + UI into a debate) is filled in during Stage 3.

use clap::Parser;
use std::path::PathBuf;

/// FUSION — multi-agent LLM debate on the command line.
#[derive(Debug, Parser)]
#[command(name = "fusion", version, about)]
pub struct Cli {
    /// One-shot query to debate. If omitted, starts an interactive chat REPL.
    #[arg(short, long)]
    pub query: Option<String>,

    /// Run the interactive onboarding wizard, then exit.
    #[arg(long)]
    pub onboard: bool,

    /// Path to an explicit config file (overrides the default location).
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Enforce scholarly "paper mode" prompt structure.
    #[arg(long)]
    pub paper_mode: bool,

    /// Number of review rounds (overrides config).
    #[arg(long)]
    pub rounds: Option<u32>,

    /// Sampling temperature (overrides config).
    #[arg(long)]
    pub temperature: Option<f32>,

    /// Max tokens per generation (overrides config).
    #[arg(long)]
    pub max_tokens: Option<u32>,

    /// JSONL run-log path (overrides config/default).
    #[arg(long)]
    pub log_file: Option<PathBuf>,

    /// Disable the progress UI (useful when piping output).
    #[arg(long)]
    pub no_progress: bool,
}

/// Parse arguments and run the requested action.
pub async fn run() -> anyhow::Result<()> {
    let _cli = Cli::parse();
    unimplemented!("implemented in Stage 3")
}
