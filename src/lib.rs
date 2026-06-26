//! FUSION — a multi-agent LLM debate engine.
//!
//! Several agents (LLMs behind OpenRouter) each answer a query, critique each
//! other's answers over N review rounds, and a synthesizer merges the refined
//! answers into one final response. See [`fusion::Fusion::debate`] for the
//! orchestration and [`prompts`] for the (verbatim-ported) prompt templates.

pub mod agent;
pub mod cli;
pub mod config;
pub mod error;
pub mod fusion;
pub mod logging;
pub mod onboarding;
pub mod openrouter;
pub mod prompts;
pub mod provider;
pub mod ui;

pub use config::Config;
pub use error::{FusionError, ProviderError, Result};
pub use fusion::{Fusion, ProgressEvent, RunMeta};
pub use provider::{ChatProvider, ChatRequest, ChatResponse, Message, Usage};
