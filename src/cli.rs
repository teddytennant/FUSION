//! Command-line interface: argument parsing and top-level dispatch.
//!
//! Wires the pieces together: load/merge config, build the OpenRouter provider,
//! construct a [`Fusion`] engine, and run either a one-shot query or an
//! interactive REPL — rendering progress via [`ProgressReporter`]. Onboarding is
//! dispatched here when `--onboard` is passed.

use crate::config::{self, Config};
use crate::fusion::{Fusion, ProgressEvent};
use crate::onboarding;
use crate::openrouter::OpenRouterClient;
use crate::provider::ChatProvider;
use crate::ui::ProgressReporter;
use anyhow::Context;
use clap::Parser;
use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::sync::Arc;

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

    /// Serve an OpenAI-compatible HTTP API instead of running a query. Point
    /// tools like Codex or opencode at `http://<host>:<port>/v1`.
    #[arg(long)]
    pub serve: bool,

    /// Host/interface to bind when `--serve` is set.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Port to bind when `--serve` is set.
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
}

impl Cli {
    /// Apply CLI flag overrides onto a loaded config (highest precedence).
    fn apply_overrides(&self, cfg: &mut Config) {
        if let Some(r) = self.rounds {
            cfg.rounds = r;
        }
        if let Some(t) = self.temperature {
            cfg.temperature = t;
        }
        if let Some(m) = self.max_tokens {
            cfg.max_tokens = m;
        }
        if let Some(ref p) = self.log_file {
            cfg.log_file = Some(p.clone());
        }
    }
}

/// Parse arguments and run the requested action.
pub async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Onboarding short-circuits everything else.
    if cli.onboard {
        let save_path = onboarding_save_path(cli.config.as_deref())?;
        onboarding::run_onboarding(&save_path).await?;
        return Ok(());
    }

    // Load config (defaults < file < env key), then layer CLI flags on top.
    let mut cfg =
        config::load_effective(cli.config.as_deref()).context("failed to load configuration")?;
    cli.apply_overrides(&mut cfg);

    // Require a key; nudge toward onboarding if missing.
    let api_key = match cfg.require_api_key() {
        Ok(k) => k.to_string(),
        Err(_) => {
            anyhow::bail!(
                "no OpenRouter API key found.\n\
                 Run `fusion --onboard` to set one up, or export OPENROUTER_API_KEY."
            );
        }
    };

    let client = OpenRouterClient::new(api_key, cfg.extra_headers.clone())
        .context("failed to build OpenRouter client")?;
    let provider: Arc<dyn ChatProvider> = Arc::new(client);
    let fusion = Fusion::from_config(&cfg, provider).context("invalid configuration")?;

    // Serve mode short-circuits the one-shot/REPL flow.
    if cli.serve {
        return crate::serve::run_server(Arc::new(fusion), &cli.host, cli.port, cli.paper_mode)
            .await
            .context("server failed");
    }

    // Progress goes to stderr; only enable it when stderr is a TTY and the user
    // didn't opt out.
    let progress_enabled = !cli.no_progress && std::io::stderr().is_terminal();

    match cli.query.clone() {
        Some(q) => {
            run_one(&fusion, &q, cli.paper_mode, progress_enabled).await?;
        }
        None => {
            run_repl(&fusion, cli.paper_mode, progress_enabled).await?;
        }
    }
    Ok(())
}

/// Run a single debate and print the final answer to stdout.
async fn run_one(
    fusion: &Fusion,
    query: &str,
    paper_mode: bool,
    progress_enabled: bool,
) -> anyhow::Result<()> {
    let mut reporter = ProgressReporter::new(progress_enabled);
    let (answer, _meta) = fusion
        .debate(query, paper_mode, &mut |e: ProgressEvent| {
            reporter.handle(&e)
        })
        .await
        .context("debate failed")?;
    // The answer is the program's real output: stdout, nothing else on it.
    println!("{answer}");
    Ok(())
}

/// Interactive REPL: each input line is debated; EOF or `exit`/`quit` ends it.
async fn run_repl(fusion: &Fusion, paper_mode: bool, progress_enabled: bool) -> anyhow::Result<()> {
    eprintln!("FUSION interactive chat. Type a query and press Enter. Ctrl-D or `exit` to quit.");
    let stdin = std::io::stdin();
    loop {
        eprint!("\n› ");
        std::io::stderr().flush().ok();
        let mut line = String::new();
        let n = stdin.read_line(&mut line).context("failed to read input")?;
        if n == 0 {
            // EOF.
            eprintln!();
            break;
        }
        let query = line.trim();
        if query.is_empty() {
            continue;
        }
        if matches!(query, "exit" | "quit") {
            break;
        }
        if let Err(e) = run_one(fusion, query, paper_mode, progress_enabled).await {
            // Keep the REPL alive on a single failed debate.
            eprintln!("error: {e:#}");
        }
    }
    Ok(())
}

/// Where onboarding should write the config: the explicit `--config` path if
/// given, else the platform default. Errors if neither is resolvable.
fn onboarding_save_path(explicit: Option<&std::path::Path>) -> anyhow::Result<PathBuf> {
    if let Some(p) = explicit {
        return Ok(p.to_path_buf());
    }
    config::default_config_path()
        .context("could not determine a config directory; pass --config <path>")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_parses_flags() {
        let cli = Cli::try_parse_from([
            "fusion",
            "--query",
            "hello",
            "--rounds",
            "2",
            "--paper-mode",
            "--no-progress",
        ])
        .unwrap();
        assert_eq!(cli.query.as_deref(), Some("hello"));
        assert_eq!(cli.rounds, Some(2));
        assert!(cli.paper_mode);
        assert!(cli.no_progress);
        assert!(!cli.onboard);
    }

    #[test]
    fn overrides_apply_in_precedence_order() {
        let cli = Cli::try_parse_from(["fusion", "--rounds", "7", "--temperature", "0.1"]).unwrap();
        let mut cfg = Config::default();
        cli.apply_overrides(&mut cfg);
        assert_eq!(cfg.rounds, 7);
        assert_eq!(cfg.temperature, 0.1);
        // Untouched flags leave defaults intact.
        assert_eq!(cfg.max_tokens, Config::default().max_tokens);
    }

    #[test]
    fn no_args_means_repl_no_onboard() {
        let cli = Cli::try_parse_from(["fusion"]).unwrap();
        assert!(cli.query.is_none());
        assert!(!cli.onboard);
    }
}
