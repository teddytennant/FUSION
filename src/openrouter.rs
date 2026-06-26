//! OpenRouter-backed [`ChatProvider`] implementation.
//!
//! NOTE: this is a Stage-0 stub that pins the public API. The real
//! implementation (reqwest client, request/response (de)serialization, key
//! validation, wiremock tests) is filled in during Stage 1.

use crate::error::{FusionError, ProviderError, Result};
use crate::provider::{ChatProvider, ChatRequest, ChatResponse};
use async_trait::async_trait;
use indexmap::IndexMap;

/// HTTP client that talks to the OpenRouter chat-completions API.
pub struct OpenRouterClient {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    extra_headers: IndexMap<String, String>,
    #[allow(dead_code)]
    base_url: String,
}

/// Default OpenRouter chat-completions endpoint.
pub const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

impl OpenRouterClient {
    /// Build a client with the given API key and extra attribution headers.
    pub fn new(api_key: impl Into<String>, extra_headers: IndexMap<String, String>) -> Result<Self> {
        let api_key = api_key.into();
        if api_key.trim().is_empty() {
            return Err(FusionError::MissingApiKey);
        }
        Ok(OpenRouterClient {
            api_key,
            extra_headers,
            base_url: DEFAULT_BASE_URL.to_string(),
        })
    }

    /// Override the base URL (used by integration tests to point at a mock server).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Validate the API key by issuing a tiny completion against the first
    /// reachable candidate model. Returns the model id that responded.
    pub async fn validate_key(&self, _candidate_models: &[&str]) -> Result<String> {
        unimplemented!("implemented in Stage 1")
    }
}

#[async_trait]
impl ChatProvider for OpenRouterClient {
    async fn chat(&self, _req: &ChatRequest) -> std::result::Result<ChatResponse, ProviderError> {
        unimplemented!("implemented in Stage 1")
    }
}
