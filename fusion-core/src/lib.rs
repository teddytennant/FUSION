//! FUSION core — the provider-agnostic multi-agent debate engine.
//!
//! Several agents (LLMs) each answer a query, critique each other's answers over
//! N review rounds, and a synthesizer merges the refined answers into one final
//! response. See [`fusion::Fusion::debate`] for the full orchestration,
//! [`fusion::Fusion::run_panel`] for just the answer/review phases (the caller
//! synthesizes), and [`prompts`] for the (verbatim-ported) prompt templates.
//!
//! The engine depends only on the [`provider::ChatProvider`] trait, never on a
//! concrete HTTP client. Production binaries (the standalone `fusion` CLI, or
//! Wizard's `FusionProvider`) implement that trait their own way; tests inject a
//! fake. No `reqwest`/`axum`/`clap` here — those live in the consuming binary.

pub mod agent;
pub mod config;
pub mod error;
pub mod fusion;
pub mod logging;
pub mod prompts;
pub mod provider;

pub use config::Config;
pub use error::{FusionError, ProviderError, Result};
pub use fusion::{Fusion, ProgressEvent, RunMeta};
pub use provider::{ChatProvider, ChatRequest, ChatResponse, Message, Usage};
