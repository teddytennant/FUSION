//! FUSION binary entry point.
//!
//! Thin wrapper: initialize tracing, parse CLI args, hand off to [`fusion::cli::run`],
//! and map any error to a non-zero exit while printing a friendly message.

use std::process::ExitCode;

#[tokio::main]
async fn main() -> ExitCode {
    // Logs go to stderr so they never collide with the progress UI / answer on
    // stdout. Controlled by RUST_LOG (default: warnings only).
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    match fusion::cli::run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
