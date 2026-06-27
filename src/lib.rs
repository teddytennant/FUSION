//! FUSION — a multi-agent LLM debate CLI.
//!
//! The debate engine lives in the [`fusion_core`] crate; it is re-exported here
//! so the binary's modules can keep referring to `crate::fusion`, `crate::config`,
//! `crate::provider`, etc. unchanged. This crate adds the OpenRouter client and
//! the user-facing surface (CLI, onboarding, server, terminal UI).

// Re-export the whole engine so `crate::{provider,agent,prompts,fusion,error,
// logging,config}` resolve to the `fusion_core` modules.
pub use fusion_core::*;

pub mod cli;
pub mod onboarding;
pub mod openrouter;
pub mod serve;
pub mod ui;
