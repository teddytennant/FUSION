//! Interactive onboarding wizard.
//!
//! Walks a first-time user through supplying an OpenRouter API key (validated
//! with a live, cheap completion), choosing the agent roster (default 2-model
//! fusion or a custom comma-separated model list), and the number of review
//! rounds, then writes the resulting [`Config`] to disk.
//!
//! All prompt/network/filesystem I/O lives in [`run_onboarding`]. The pure
//! config-assembly logic is factored into [`build_config`] so it can be unit
//! tested without a TTY or network.

use crate::config::{AgentConfig, Config, SynthesizerConfig};
use crate::error::{FusionError, Result};
use crate::openrouter::OpenRouterClient;
use dialoguer::{Confirm, Input, Password};
use indexmap::IndexMap;
use std::path::Path;

/// Cheap candidate models used to validate the supplied key during onboarding.
const CANDIDATES: [&str; 3] = [
    "anthropic/claude-sonnet-4.6",
    "z-ai/glm-4.6",
    "openai/gpt-5.1",
];

/// Map a `dialoguer` interaction error into a [`FusionError`].
fn prompt_err(e: dialoguer::Error) -> FusionError {
    FusionError::Config(format!("onboarding prompt failed: {e}"))
}

/// Build a [`Config`] from already-collected onboarding inputs.
///
/// Pure: performs no prompts, no network calls, and no filesystem access. The
/// caller is responsible for collecting `api_key` (validated or not),
/// `custom_models`, and `rounds` before invoking this.
///
/// - `custom_models = None` (or an effectively empty list) → the default
///   2-model roster and its synthesizer are retained.
/// - `custom_models = Some(models)` with at least one non-blank id → one
///   [`AgentConfig`] per model (name = model id, no role, no fallbacks), with
///   the first model used as the synthesizer.
/// - `rounds` is clamped to a minimum of 1.
fn build_config(
    api_key: String,
    custom_models: Option<Vec<String>>,
    rounds: u32,
) -> Result<Config> {
    let defaults = Config::default();

    // Normalize any custom roster: trim entries and drop blanks.
    let custom: Vec<String> = custom_models
        .unwrap_or_default()
        .into_iter()
        .map(|m| m.trim().to_string())
        .filter(|m| !m.is_empty())
        .collect();

    // Either a custom roster, or fall back to the defaults if none was given.
    let (agents, synthesizer) = if custom.is_empty() {
        (defaults.agents, defaults.synthesizer)
    } else {
        let agents = custom
            .iter()
            .map(|model| AgentConfig {
                name: model.clone(),
                model: model.clone(),
                role: None,
                fallback_models: Vec::new(),
            })
            .collect();
        // The first custom model doubles as the synthesizer.
        let synth_model = custom[0].clone();
        let synthesizer = SynthesizerConfig {
            name: synth_model.clone(),
            model: synth_model,
            fallback_models: Vec::new(),
        };
        (agents, synthesizer)
    };

    let config = Config {
        api_key: Some(api_key),
        agents,
        synthesizer,
        rounds: rounds.max(1),
        ..Config::default()
    };

    config.validate()?;
    Ok(config)
}

/// Run the interactive onboarding wizard, persisting the result to `save_path`
/// and returning the populated [`Config`] for immediate use.
pub async fn run_onboarding(save_path: &Path) -> Result<Config> {
    print_banner();

    // 1. API key (no echo).
    let key: String = Password::new()
        .with_prompt("OpenRouter API key")
        .interact()
        .map_err(prompt_err)?;
    let key = key.trim().to_string();
    if key.is_empty() {
        return Err(FusionError::MissingApiKey);
    }

    // 2. Validate the key with a live, cheap completion.
    let client = OpenRouterClient::new(key.clone(), IndexMap::new())?;
    match client.validate_key(&CANDIDATES).await {
        Ok(model) => {
            println!("\u{2713} Key validated via {model}");
        }
        Err(e) => {
            println!("\u{2717} Key validation failed: {e}");
            let proceed = Confirm::new()
                .with_prompt("Save the key anyway and continue?")
                .default(false)
                .interact()
                .map_err(prompt_err)?;
            if !proceed {
                return Err(e);
            }
        }
    }

    // 3. Roster: default 2-model fusion or custom.
    let use_default = Confirm::new()
        .with_prompt("Use the default 2-model fusion (Claude Opus 4.8 + GLM 5.2)?")
        .default(true)
        .interact()
        .map_err(prompt_err)?;

    let custom_models = if use_default {
        None
    } else {
        let raw: String = Input::new()
            .with_prompt("Comma-separated OpenRouter model ids")
            .allow_empty(true)
            .interact_text()
            .map_err(prompt_err)?;
        let models: Vec<String> = raw
            .split(',')
            .map(|m| m.trim().to_string())
            .filter(|m| !m.is_empty())
            .collect();
        if models.is_empty() {
            println!("No models entered; falling back to the default roster.");
            None
        } else {
            Some(models)
        }
    };

    // 4. Review rounds (>= 1).
    let rounds: u32 = Input::new()
        .with_prompt("Number of review rounds")
        .default(3u32)
        .validate_with(|n: &u32| -> std::result::Result<(), &str> {
            if *n >= 1 {
                Ok(())
            } else {
                Err("rounds must be at least 1")
            }
        })
        .interact_text()
        .map_err(prompt_err)?;

    // 5. Assemble, validate, and persist.
    let config = build_config(key, custom_models, rounds)?;
    config.save_to_path(save_path)?;
    println!("Configuration saved to {}", save_path.display());

    Ok(config)
}

/// Print a short ASCII banner to stdout.
fn print_banner() {
    println!();
    println!(" ███████╗██╗   ██╗███████╗██╗ ██████╗ ███╗   ██╗");
    println!(" ██╔════╝██║   ██║██╔════╝██║██╔═══██╗████╗  ██║");
    println!(" █████╗  ██║   ██║███████╗██║██║   ██║██╔██╗ ██║");
    println!(" ██╔══╝  ██║   ██║╚════██║██║██║   ██║██║╚██╗██║");
    println!(" ██║     ╚██████╔╝███████║██║╚██████╔╝██║ ╚████║");
    println!(" ╚═╝      ╚═════╝ ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝");
    println!("   multi-agent LLM debate \u{2014} let's get you set up");
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_roster_path() {
        let cfg = build_config("sk-test".into(), None, 5).expect("builds");
        assert_eq!(cfg.agents.len(), 2);
        assert_eq!(cfg.api_key.as_deref(), Some("sk-test"));
        assert_eq!(cfg.rounds, 5);
        cfg.validate().expect("valid");
    }

    #[test]
    fn custom_roster_path() {
        let cfg = build_config("sk-test".into(), Some(vec!["a/b".into(), "c/d".into()]), 2)
            .expect("builds");
        assert_eq!(cfg.agents.len(), 2);
        assert_eq!(cfg.agents[0].name, "a/b");
        assert_eq!(cfg.agents[0].model, "a/b");
        assert_eq!(cfg.agents[1].name, "c/d");
        assert_eq!(cfg.synthesizer.model, "a/b");
        assert_eq!(cfg.rounds, 2);
        cfg.validate().expect("valid");
    }

    #[test]
    fn empty_custom_falls_back_to_default() {
        let cfg = build_config("sk-test".into(), Some(Vec::new()), 3).expect("builds");
        assert_eq!(cfg.agents.len(), 2);
        // Equal rosters to the default config.
        assert_eq!(cfg.agents, Config::default().agents);
    }

    #[test]
    fn custom_with_only_blanks_falls_back_to_default() {
        let cfg =
            build_config("sk-test".into(), Some(vec!["  ".into(), "".into()]), 3).expect("builds");
        assert_eq!(cfg.agents.len(), 2);
    }

    #[test]
    fn rounds_clamped_to_at_least_one() {
        let cfg = build_config("sk-test".into(), None, 0).expect("builds");
        assert_eq!(cfg.rounds, 1);
    }

    #[test]
    fn build_config_round_trips_through_disk() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let cfg = build_config("sk-test".into(), None, 4).expect("builds");
        cfg.save_to_path(&path).expect("save");
        let loaded = Config::load_from_path(&path).expect("load");
        assert_eq!(cfg, loaded);
    }
}
