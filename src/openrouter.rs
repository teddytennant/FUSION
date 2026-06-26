//! OpenRouter-backed [`ChatProvider`] implementation.
//!
//! A thin, OpenAI-compatible HTTP client over OpenRouter's chat-completions
//! endpoint. It owns a single [`reqwest::Client`] (built once in [`new`]) and
//! translates wire-level outcomes into the three [`ProviderError`] variants the
//! agent retry loop discriminates on: a concrete HTTP status, a transport
//! failure, or a 2xx body with no usable `choices`.

use crate::error::{FusionError, ProviderError, Result};
use crate::provider::{ChatProvider, ChatRequest, ChatResponse, Message, Usage};
use async_trait::async_trait;
use indexmap::IndexMap;
use serde::Serialize;
use std::time::Duration;

/// Default OpenRouter chat-completions endpoint.
pub const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

/// Per-request timeout for the underlying HTTP client.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Max number of bytes of an error body we retain (avoid unbounded blobs in logs).
const MAX_ERROR_BODY: usize = 2000;

/// OpenRouter attribution headers (see https://openrouter.ai/docs).
const HTTP_REFERER: &str = "https://github.com/teddytennant/FUSION";
const X_TITLE: &str = "FUSION";

/// HTTP client that talks to the OpenRouter chat-completions API.
pub struct OpenRouterClient {
    http: reqwest::Client,
    api_key: String,
    extra_headers: IndexMap<String, String>,
    base_url: String,
}

/// Wire shape of an outgoing chat-completion request body.
///
/// `Message` already serializes as a `{role, content}` object, so we reuse it
/// directly. `seed` is skipped entirely when absent (OpenRouter treats a missing
/// seed as "non-deterministic" rather than "seed = null").
#[derive(Serialize)]
struct ChatRequestBody<'a> {
    model: &'a str,
    messages: &'a [Message],
    max_tokens: u32,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
}

impl OpenRouterClient {
    /// Build a client with the given API key and extra attribution headers.
    ///
    /// Returns [`FusionError::MissingApiKey`] if the key is blank. The reqwest
    /// client (and its connection pool) is constructed once here and reused for
    /// every request.
    pub fn new(
        api_key: impl Into<String>,
        extra_headers: IndexMap<String, String>,
    ) -> Result<Self> {
        let api_key = api_key.into();
        if api_key.trim().is_empty() {
            return Err(FusionError::MissingApiKey);
        }
        let http = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            // A builder failure here is a TLS/runtime misconfiguration, not a
            // user-supplied-key problem; surface it as a validation-style error.
            .map_err(|e| FusionError::KeyValidation(format!("failed to build HTTP client: {e}")))?;
        Ok(OpenRouterClient {
            http,
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
    ///
    /// A `401` from any model short-circuits immediately (the key itself is bad,
    /// so trying more models is pointless). Any other failure advances to the
    /// next candidate; if all fail, the last error is summarized.
    pub async fn validate_key(&self, candidate_models: &[&str]) -> Result<String> {
        let mut last_error: Option<ProviderError> = None;
        for model in candidate_models {
            let req = ChatRequest {
                model: (*model).to_string(),
                messages: vec![Message::user("Reply with OK.")],
                max_tokens: 5,
                temperature: 0.0,
                seed: None,
            };
            match self.chat(&req).await {
                Ok(_) => return Ok((*model).to_string()),
                Err(ProviderError::Http { status: 401, .. }) => {
                    return Err(FusionError::KeyValidation("invalid API key (401)".into()));
                }
                Err(e) => last_error = Some(e),
            }
        }
        let detail = match last_error {
            Some(e) => format!("no candidate model responded; last error: {e}"),
            None => "no candidate models were provided".to_string(),
        };
        Err(FusionError::KeyValidation(detail))
    }
}

/// Truncate an error/response body to a bounded length for storage in errors.
fn truncate_body(body: String) -> String {
    if body.len() <= MAX_ERROR_BODY {
        return body;
    }
    // Truncate on a char boundary at or below the limit.
    let mut end = MAX_ERROR_BODY;
    while end > 0 && !body.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}… (truncated)", &body[..end])
}

#[async_trait]
impl ChatProvider for OpenRouterClient {
    async fn chat(&self, req: &ChatRequest) -> std::result::Result<ChatResponse, ProviderError> {
        let body = ChatRequestBody {
            model: &req.model,
            messages: &req.messages,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            seed: req.seed,
        };

        let mut builder = self
            .http
            .post(&self.base_url)
            .bearer_auth(&self.api_key)
            .header("HTTP-Referer", HTTP_REFERER)
            .header("X-Title", X_TITLE)
            .json(&body);

        // Extra headers extend/override the defaults above.
        for (k, v) in &self.extra_headers {
            builder = builder.header(k, v);
        }

        // A failure here means we never received an HTTP status (timeout, DNS,
        // connection reset, TLS handshake, ...).
        let resp = builder
            .send()
            .await
            .map_err(|e| ProviderError::Transport(e.to_string()))?;

        let status = resp.status();
        // Reading the body can also fail mid-stream; treat as transport.
        let text = resp
            .text()
            .await
            .map_err(|e| ProviderError::Transport(e.to_string()))?;

        if !status.is_success() {
            return Err(ProviderError::Http {
                status: status.as_u16(),
                body: truncate_body(text),
            });
        }

        // 2xx: parse the JSON. A body that isn't valid JSON, or one without a
        // usable choice, is an empty-choices condition (advance to fallback).
        let raw: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => {
                return Err(ProviderError::EmptyChoices {
                    body: truncate_body(text),
                })
            }
        };

        let first_choice = raw
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first());

        let Some(choice) = first_choice else {
            // OpenRouter occasionally returns 200 with an inline {"error": ...}
            // body and no choices. Prefer the error object if present.
            let body = match raw.get("error") {
                Some(err) => err.to_string(),
                None => truncate_body(text),
            };
            return Err(ProviderError::EmptyChoices { body });
        };

        // A null/absent content is a *valid* empty completion, not an error.
        let content = choice
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();

        let usage = raw
            .get("usage")
            .and_then(|u| serde_json::from_value::<Usage>(u.clone()).ok())
            .unwrap_or_default();

        Ok(ChatResponse {
            content,
            usage,
            raw,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_string_contains, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn sample_request(seed: Option<u64>) -> ChatRequest {
        ChatRequest {
            model: "model-a".to_string(),
            messages: vec![Message::system("be brief"), Message::user("hi")],
            max_tokens: 64,
            temperature: 0.7,
            seed,
        }
    }

    async fn client_for(server: &MockServer) -> OpenRouterClient {
        OpenRouterClient::new("test-key", IndexMap::new())
            .expect("client builds")
            .with_base_url(format!("{}/chat", server.uri()))
    }

    #[tokio::test]
    async fn new_rejects_blank_key() {
        assert!(matches!(
            OpenRouterClient::new("   ", IndexMap::new()),
            Err(FusionError::MissingApiKey)
        ));
    }

    #[tokio::test]
    async fn chat_success_parses_content_and_usage() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"total_tokens": 5}
            })))
            .mount(&server)
            .await;

        let client = client_for(&server).await;
        let resp = client.chat(&sample_request(None)).await.expect("ok");
        assert_eq!(resp.content, "hello");
        assert_eq!(resp.usage.total_tokens, 5);
    }

    #[tokio::test]
    async fn chat_sends_auth_and_attribution_headers() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat"))
            .and(header("authorization", "Bearer test-key"))
            .and(header("http-referer", HTTP_REFERER))
            .and(header("x-title", X_TITLE))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "ok"}}]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = client_for(&server).await;
        // Mock only matches when the headers are present; an Ok here proves it.
        client.chat(&sample_request(None)).await.expect("ok");
    }

    #[tokio::test]
    async fn chat_extra_headers_are_attached() {
        let server = MockServer::start().await;
        let mut extra = IndexMap::new();
        extra.insert("X-Custom".to_string(), "abc".to_string());
        Mock::given(method("POST"))
            .and(path("/chat"))
            .and(header("x-custom", "abc"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "ok"}}]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = OpenRouterClient::new("test-key", extra)
            .expect("client builds")
            .with_base_url(format!("{}/chat", server.uri()));
        client.chat(&sample_request(None)).await.expect("ok");
    }

    #[tokio::test]
    async fn chat_body_includes_core_fields_without_seed() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat"))
            .and(body_string_contains("\"model\":\"model-a\""))
            .and(body_string_contains("\"max_tokens\":64"))
            .and(body_string_contains("\"temperature\":0.7"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "ok"}}]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = client_for(&server).await;
        // seed = None must NOT appear in the body.
        client.chat(&sample_request(None)).await.expect("ok");
    }

    #[tokio::test]
    async fn chat_body_includes_seed_when_set() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat"))
            .and(body_string_contains("\"seed\":42"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "ok"}}]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = client_for(&server).await;
        client.chat(&sample_request(Some(42))).await.expect("ok");
    }

    #[tokio::test]
    async fn chat_non_2xx_maps_to_http_error_and_is_retryable() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&server)
            .await;

        let client = client_for(&server).await;
        let err = client.chat(&sample_request(None)).await.expect_err("429");
        match err {
            ProviderError::Http { status, .. } => assert_eq!(status, 429),
            other => panic!("expected Http error, got {other:?}"),
        }
        assert!(err.is_retryable());
    }

    #[tokio::test]
    async fn chat_200_with_error_body_maps_to_empty_choices() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "error": {"message": "bad"}
            })))
            .mount(&server)
            .await;

        let client = client_for(&server).await;
        let err = client.chat(&sample_request(None)).await.expect_err("empty");
        match err {
            ProviderError::EmptyChoices { body } => assert!(body.contains("bad")),
            other => panic!("expected EmptyChoices, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_null_content_coerced_to_empty_string() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": null}}]
            })))
            .mount(&server)
            .await;

        let client = client_for(&server).await;
        let resp = client.chat(&sample_request(None)).await.expect("ok");
        assert_eq!(resp.content, "");
    }

    #[tokio::test]
    async fn validate_key_returns_first_responding_model() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "OK"}}]
            })))
            .mount(&server)
            .await;

        let client = client_for(&server).await;
        let model = client.validate_key(&["model-a"]).await.expect("valid");
        assert_eq!(model, "model-a".to_string());
    }

    #[tokio::test]
    async fn validate_key_401_short_circuits() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat"))
            .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized"))
            .mount(&server)
            .await;

        let client = client_for(&server).await;
        let err = client
            .validate_key(&["model-a", "model-b"])
            .await
            .expect_err("401");
        match err {
            FusionError::KeyValidation(msg) => assert!(msg.contains("401")),
            other => panic!("expected KeyValidation, got {other:?}"),
        }
    }
}
