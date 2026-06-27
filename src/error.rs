//! Error types for the FUSION library.
//!
//! Two layers:
//! - [`ProviderError`] is the low-level transport/HTTP error surfaced by a
//!   [`crate::provider::ChatProvider`]. It carries enough information to decide
//!   whether a request is worth retrying.
//! - [`FusionError`] is the library-level error returned by orchestration code
//!   (agents, the fusion engine, config loading). The CLI maps this to a process
//!   exit at the boundary via `anyhow`.

use thiserror::Error;

/// Result alias for library-level operations.
pub type Result<T> = std::result::Result<T, FusionError>;

/// Low-level error from a chat provider call.
///
/// Variants distinguish the cases the retry loop cares about: a concrete HTTP
/// status (which may or may not be retryable), a transport-layer failure
/// (timeouts, DNS, connection resets — always retryable), and a structurally
/// empty/erroring success body (e.g. OpenRouter returning HTTP 200 with an
/// `{"error": ...}` payload and no `choices`).
#[derive(Debug, Error, Clone)]
pub enum ProviderError {
    /// The server returned a non-2xx HTTP status.
    #[error("HTTP {status}: {body}")]
    Http { status: u16, body: String },

    /// A transport-level failure occurred before a status was received
    /// (timeout, connection reset, TLS error, DNS failure, ...).
    #[error("transport error: {0}")]
    Transport(String),

    /// The response was 2xx but contained no usable choices (e.g. a 200 body
    /// carrying an inline error object).
    #[error("no choices returned: {body}")]
    EmptyChoices { body: String },
}

impl ProviderError {
    /// Whether this error is worth retrying against the *same* model.
    ///
    /// Mirrors the Python implementation: retry transient HTTP statuses
    /// (`408`, `409`, `429`, and all `5xx`) and any transport failure. Every
    /// other HTTP status (notably `4xx` such as `400`, `401`, `404`) is
    /// non-retryable and should advance to the next fallback model instead.
    /// An empty-choices body is treated as non-retryable (advance to fallback).
    pub fn is_retryable(&self) -> bool {
        match self {
            ProviderError::Http { status, .. } => {
                matches!(status, 408 | 409 | 429) || (500..600).contains(status)
            }
            ProviderError::Transport(_) => true,
            ProviderError::EmptyChoices { .. } => false,
        }
    }
}

/// Library-level error type for orchestration and configuration.
#[derive(Debug, Error)]
pub enum FusionError {
    /// No OpenRouter API key was available from config or environment.
    #[error("no OpenRouter API key found; set OPENROUTER_API_KEY or run `fusion --onboard`")]
    MissingApiKey,

    /// A configuration value was invalid.
    #[error("config error: {0}")]
    Config(String),

    /// Two or more agents share the same name, which would corrupt the
    /// name-keyed debate bookkeeping.
    #[error("duplicate agent names are not allowed: {0:?}")]
    DuplicateAgentNames(Vec<String>),

    /// The configuration contained no agents.
    #[error("at least one agent must be configured")]
    NoAgents,

    /// Every model (primary + fallbacks) failed for a given agent.
    #[error("agent '{agent}' exhausted all models; last error: {last_error}")]
    AllModelsFailed {
        agent: String,
        last_error: ProviderError,
    },

    /// The provided API key failed validation during onboarding.
    #[error("API key validation failed: {0}")]
    KeyValidation(String),

    /// A filesystem error occurred (reading/writing config or logs).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// A TOML (de)serialization error occurred.
    #[error("config (de)serialization error: {0}")]
    Toml(String),
}

impl From<toml::de::Error> for FusionError {
    fn from(e: toml::de::Error) -> Self {
        FusionError::Toml(e.to_string())
    }
}

impl From<toml::ser::Error> for FusionError {
    fn from(e: toml::ser::Error) -> Self {
        FusionError::Toml(e.to_string())
    }
}
