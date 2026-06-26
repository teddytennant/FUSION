//! Configuration: schema, defaults, file/env/flag merge, and on-disk paths.
//!
//! Precedence (low → high): built-in defaults < config file < `OPENROUTER_API_KEY`
//! environment variable (api_key only) < CLI flags. The config file is TOML,
//! stored at the platform config dir (`~/.config/fusion/config.toml` on Linux)
//! resolved via the `directories` crate.

use crate::error::{FusionError, Result};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Environment variable that supplies the OpenRouter API key.
pub const API_KEY_ENV: &str = "OPENROUTER_API_KEY";

/// Configuration for a single debating agent.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentConfig {
    /// Human-readable, unique name (used as the key in debate bookkeeping).
    pub name: String,
    /// Primary OpenRouter model id (e.g. `google/gemini-2.5-pro`).
    pub model: String,
    /// Optional role description, woven into the agent's system prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Models tried, in order, if the primary fails with a non-retryable error.
    #[serde(default)]
    pub fallback_models: Vec<String>,
}

/// Configuration for the synthesizer agent that merges the final answers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SynthesizerConfig {
    pub name: String,
    pub model: String,
    #[serde(default)]
    pub fallback_models: Vec<String>,
}

/// Top-level configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Config {
    /// OpenRouter API key. May be absent in the file (supplied via env instead).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    pub agents: Vec<AgentConfig>,
    pub synthesizer: SynthesizerConfig,
    pub rounds: u32,
    pub max_tokens: u32,
    pub temperature: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Path for the JSONL run log. `None` → the default under the data dir.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub log_file: Option<PathBuf>,
    /// Extra HTTP headers attached to every OpenRouter request (e.g. attribution).
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub extra_headers: IndexMap<String, String>,
}

impl Default for Config {
    /// The built-in default roster, ported from the original Python
    /// `build_default_config`, modernized to OpenRouter-only model ids.
    fn default() -> Self {
        Config {
            api_key: None,
            agents: vec![
                AgentConfig {
                    name: "Gemini 2.5 Pro".into(),
                    model: "google/gemini-2.5-pro".into(),
                    role: Some("general reasoning".into()),
                    fallback_models: vec![
                        "google/gemini-2.5-flash".into(),
                        "google/gemini-2.0-flash-001".into(),
                    ],
                },
                AgentConfig {
                    name: "Grok-4".into(),
                    model: "x-ai/grok-4".into(),
                    role: Some("factual accuracy".into()),
                    fallback_models: vec!["x-ai/grok-3".into(), "x-ai/grok-3-mini".into()],
                },
                AgentConfig {
                    name: "DeepSeek".into(),
                    model: "deepseek/deepseek-chat".into(),
                    role: Some("coding and math".into()),
                    fallback_models: vec!["deepseek/deepseek-r1".into()],
                },
            ],
            synthesizer: SynthesizerConfig {
                name: "Gemini 2.5 Pro".into(),
                model: "google/gemini-2.5-pro".into(),
                fallback_models: vec![
                    "google/gemini-2.5-flash".into(),
                    "google/gemini-2.0-flash-001".into(),
                ],
            },
            rounds: 3,
            max_tokens: 1000,
            temperature: 0.7,
            seed: None,
            log_file: None,
            extra_headers: IndexMap::new(),
        }
    }
}

impl Config {
    /// Validate structural invariants the orchestrator relies on.
    pub fn validate(&self) -> Result<()> {
        if self.agents.is_empty() {
            return Err(FusionError::NoAgents);
        }
        let mut seen = std::collections::HashSet::new();
        let mut dups = Vec::new();
        for a in &self.agents {
            if !seen.insert(a.name.as_str()) {
                dups.push(a.name.clone());
            }
        }
        if !dups.is_empty() {
            return Err(FusionError::DuplicateAgentNames(dups));
        }
        Ok(())
    }

    /// Parse a [`Config`] from a TOML string.
    pub fn from_toml_str(s: &str) -> Result<Self> {
        let cfg: Config = toml::from_str(s)?;
        Ok(cfg)
    }

    /// Serialize this [`Config`] to a pretty TOML string.
    pub fn to_toml_string(&self) -> Result<String> {
        Ok(toml::to_string_pretty(self)?)
    }

    /// Load a [`Config`] from a TOML file on disk.
    pub fn load_from_path(path: &Path) -> Result<Self> {
        let s = std::fs::read_to_string(path)?;
        Self::from_toml_str(&s)
    }

    /// Write this [`Config`] to `path`, creating parent directories as needed.
    pub fn save_to_path(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let s = self.to_toml_string()?;
        std::fs::write(path, s)?;
        Ok(())
    }

    /// Apply the `OPENROUTER_API_KEY` environment variable, if set and the
    /// config does not already carry a (non-empty) key. Env overrides an empty
    /// file value but a present file value already loaded takes the file's key
    /// only when env is unset — matching the documented precedence
    /// (file < env), env wins when present.
    pub fn apply_env_key(&mut self) {
        if let Ok(key) = std::env::var(API_KEY_ENV) {
            let key = key.trim().to_string();
            if !key.is_empty() {
                self.api_key = Some(key);
            }
        }
    }

    /// Resolve the effective API key, returning an error if none is available.
    pub fn require_api_key(&self) -> Result<&str> {
        match self.api_key.as_deref() {
            Some(k) if !k.trim().is_empty() => Ok(k),
            _ => Err(FusionError::MissingApiKey),
        }
    }
}

/// The default on-disk path for the config file
/// (`~/.config/fusion/config.toml` on Linux), if a home/config dir is resolvable.
pub fn default_config_path() -> Option<PathBuf> {
    directories::ProjectDirs::from("", "", "fusion").map(|d| d.config_dir().join("config.toml"))
}

/// The default on-disk path for the JSONL run log
/// (`~/.local/share/fusion/runs.jsonl` on Linux).
pub fn default_log_path() -> Option<PathBuf> {
    directories::ProjectDirs::from("", "", "fusion").map(|d| d.data_dir().join("runs.jsonl"))
}

/// Load the effective config given an optional explicit `--config` path.
///
/// Resolution: if `explicit` is given, load it; else load the default path if it
/// exists; else fall back to [`Config::default`]. The environment API key is
/// then layered on top. CLI-flag overrides are applied by the caller.
pub fn load_effective(explicit: Option<&Path>) -> Result<Config> {
    let mut cfg = match explicit {
        Some(p) => Config::load_from_path(p)?,
        None => match default_config_path() {
            Some(p) if p.exists() => Config::load_from_path(&p)?,
            _ => Config::default(),
        },
    };
    cfg.apply_env_key();
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid_and_openrouter_only() {
        let cfg = Config::default();
        cfg.validate().unwrap();
        assert_eq!(cfg.agents.len(), 3);
        // Every model id is an OpenRouter slash-prefixed id.
        for a in &cfg.agents {
            assert!(
                a.model.contains('/'),
                "model {} not slash-prefixed",
                a.model
            );
        }
    }

    #[test]
    fn validate_rejects_empty_and_duplicate() {
        let mut cfg = Config::default();
        cfg.agents.clear();
        assert!(matches!(cfg.validate(), Err(FusionError::NoAgents)));

        let mut cfg = Config::default();
        cfg.agents[1].name = cfg.agents[0].name.clone();
        assert!(matches!(
            cfg.validate(),
            Err(FusionError::DuplicateAgentNames(_))
        ));
    }

    #[test]
    fn toml_round_trips() {
        let cfg = Config {
            api_key: Some("secret".into()),
            ..Config::default()
        };
        let s = cfg.to_toml_string().unwrap();
        let back = Config::from_toml_str(&s).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn require_api_key_errors_when_blank() {
        let mut cfg = Config::default();
        assert!(matches!(
            cfg.require_api_key(),
            Err(FusionError::MissingApiKey)
        ));
        cfg.api_key = Some("   ".into());
        assert!(matches!(
            cfg.require_api_key(),
            Err(FusionError::MissingApiKey)
        ));
        cfg.api_key = Some("real".into());
        assert_eq!(cfg.require_api_key().unwrap(), "real");
    }

    #[test]
    fn save_and_load_from_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("config.toml");
        let cfg = Config::default();
        cfg.save_to_path(&path).unwrap();
        let loaded = Config::load_from_path(&path).unwrap();
        assert_eq!(cfg, loaded);
    }
}
