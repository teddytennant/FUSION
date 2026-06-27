//! The chat-provider abstraction.
//!
//! The orchestrator ([`crate::agent::Agent`], [`crate::fusion::Fusion`]) depends
//! only on the [`ChatProvider`] trait, never on a concrete HTTP client. This is
//! the seam that keeps the whole debate testable without a network: production
//! uses [`crate::openrouter::OpenRouterClient`]; tests inject a fake.

use crate::error::ProviderError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// A single chat message in the OpenAI-compatible format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Message {
            role: "system".into(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Message {
            role: "user".into(),
            content: content.into(),
        }
    }
}

/// A provider-agnostic chat completion request.
///
/// `model` is the OpenRouter model id to target for *this* attempt (the agent's
/// retry/fallback loop sets it per attempt). Extra headers (e.g. the OpenRouter
/// `HTTP-Referer`/`X-Title` attribution headers) are carried alongside so the
/// concrete client can attach them.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub seed: Option<u64>,
}

/// Token-usage accounting as reported by the provider (best-effort).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
}

/// A normalized successful chat completion.
///
/// `content` is coerced to a (possibly empty) string by the client so callers
/// never have to handle a null message body. `raw` preserves the full provider
/// JSON for the JSONL run log.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub content: String,
    pub usage: Usage,
    pub raw: serde_json::Value,
}

/// The abstraction every consumer of model output depends on.
///
/// Implementations must be `Send + Sync` so the orchestrator can hold one behind
/// an `Arc<dyn ChatProvider>` and share it across agents.
#[async_trait]
pub trait ChatProvider: Send + Sync {
    /// Perform a single chat completion attempt for `req.model`.
    ///
    /// This is one *attempt*: retry/backoff and fallback-model selection live in
    /// [`crate::agent::Agent`], not here. Implementations return
    /// [`ProviderError`] so the agent can decide whether to retry or advance.
    async fn chat(&self, req: &ChatRequest) -> std::result::Result<ChatResponse, ProviderError>;
}
