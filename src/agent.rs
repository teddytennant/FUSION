//! A single debating agent: wraps a [`ChatProvider`] with retry, exponential
//! backoff, and fallback-model selection.
//!
//! Ported from the Python `Agent.generate`. One generation tries the primary
//! model then each fallback in order. Within a model, retryable failures
//! (`408/409/429/5xx`, transport) back off exponentially (1s → 2s → 4s by
//! default) up to `max_retries`; a non-retryable failure (other `4xx`, empty
//! choices) advances immediately to the next model. If every model is exhausted,
//! [`FusionError::AllModelsFailed`] is returned — there is no mock fallback.

use crate::error::{FusionError, ProviderError};
use crate::provider::{ChatProvider, ChatRequest, ChatResponse, Message};
use std::sync::Arc;
use std::time::Duration;

/// A debating agent bound to a provider and a model (+ fallbacks).
pub struct Agent {
    pub name: String,
    pub model: String,
    pub fallback_models: Vec<String>,
    pub system_prompt: String,
    provider: Arc<dyn ChatProvider>,
    max_retries: u32,
    base_backoff: Duration,
}

impl Agent {
    /// Create an agent. Defaults: 3 retries, 1s base backoff (matching Python).
    pub fn new(
        name: impl Into<String>,
        model: impl Into<String>,
        fallback_models: Vec<String>,
        system_prompt: impl Into<String>,
        provider: Arc<dyn ChatProvider>,
    ) -> Self {
        Agent {
            name: name.into(),
            model: model.into(),
            fallback_models,
            system_prompt: system_prompt.into(),
            provider,
            max_retries: 3,
            base_backoff: Duration::from_secs(1),
        }
    }

    /// Override the per-model retry count (default 3).
    pub fn with_max_retries(mut self, n: u32) -> Self {
        self.max_retries = n.max(1);
        self
    }

    /// Override the base backoff (default 1s). Tests use a paused clock so the
    /// real value is irrelevant; this exists for explicitness.
    pub fn with_base_backoff(mut self, d: Duration) -> Self {
        self.base_backoff = d;
        self
    }

    /// Generate a response to `prompt`, trying the primary then fallback models.
    pub async fn generate(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        seed: Option<u64>,
    ) -> Result<ChatResponse, FusionError> {
        let messages = vec![
            Message::system(self.system_prompt.clone()),
            Message::user(prompt.to_string()),
        ];

        let mut models = Vec::with_capacity(1 + self.fallback_models.len());
        models.push(self.model.clone());
        models.extend(self.fallback_models.iter().cloned());

        let mut last_err = ProviderError::Transport("no attempt was made".into());

        for model in models {
            let req = ChatRequest {
                model: model.clone(),
                messages: messages.clone(),
                max_tokens,
                temperature,
                seed,
            };
            let mut backoff = self.base_backoff;
            for attempt in 1..=self.max_retries {
                match self.provider.chat(&req).await {
                    Ok(resp) => return Ok(resp),
                    Err(e) => {
                        let retryable = e.is_retryable();
                        last_err = e;
                        if retryable && attempt < self.max_retries {
                            tracing::warn!(
                                "{} model {} attempt {} failed ({}); retrying in {:?}",
                                self.name,
                                model,
                                attempt,
                                last_err,
                                backoff
                            );
                            tokio::time::sleep(backoff).await;
                            backoff *= 2;
                            continue;
                        }
                        // Non-retryable, or retries exhausted: advance to the
                        // next fallback model.
                        break;
                    }
                }
            }
        }

        Err(FusionError::AllModelsFailed {
            agent: self.name.clone(),
            last_error: last_err,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::Usage;
    use async_trait::async_trait;
    use std::sync::Mutex;

    /// A scripted, no-network provider for tests. Each `chat` call pops the next
    /// scripted result; requests are captured for assertions.
    pub struct FakeProvider {
        pub script: Mutex<std::collections::VecDeque<Result<ChatResponse, ProviderError>>>,
        pub calls: Mutex<Vec<ChatRequest>>,
    }

    impl FakeProvider {
        fn new(script: Vec<Result<ChatResponse, ProviderError>>) -> Arc<Self> {
            Arc::new(FakeProvider {
                script: Mutex::new(script.into_iter().collect()),
                calls: Mutex::new(Vec::new()),
            })
        }
        fn ok(content: &str) -> ChatResponse {
            ChatResponse {
                content: content.to_string(),
                usage: Usage::default(),
                raw: serde_json::json!({"choices":[{"message":{"content":content}}]}),
            }
        }
    }

    #[async_trait]
    impl ChatProvider for FakeProvider {
        async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
            self.calls.lock().unwrap().push(req.clone());
            self.script
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or(Err(ProviderError::Transport("script exhausted".into())))
        }
    }

    #[tokio::test(start_paused = true)]
    async fn retries_then_succeeds_on_same_model() {
        let provider = FakeProvider::new(vec![
            Err(ProviderError::Http {
                status: 429,
                body: "rate limited".into(),
            }),
            Ok(FakeProvider::ok("recovered")),
        ]);
        let agent = Agent::new("A", "m/primary", vec![], "sys", provider.clone());
        let res = agent.generate("q", 100, 0.7, None).await.unwrap();
        assert_eq!(res.content, "recovered");
        // Both attempts hit the *same* (primary) model.
        let calls = provider.calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert!(calls.iter().all(|c| c.model == "m/primary"));
    }

    #[tokio::test(start_paused = true)]
    async fn non_retryable_advances_to_fallback() {
        let provider = FakeProvider::new(vec![
            Err(ProviderError::Http {
                status: 404,
                body: "no such model".into(),
            }),
            Ok(FakeProvider::ok("from fallback")),
        ]);
        let agent = Agent::new(
            "A",
            "m/primary",
            vec!["m/fallback".into()],
            "sys",
            provider.clone(),
        );
        let res = agent.generate("q", 100, 0.7, None).await.unwrap();
        assert_eq!(res.content, "from fallback");
        let calls = provider.calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].model, "m/primary");
        assert_eq!(calls[1].model, "m/fallback");
    }

    #[tokio::test(start_paused = true)]
    async fn exhausting_all_models_errors() {
        // primary: 3 retryable failures; fallback: 3 retryable failures.
        let mut script = Vec::new();
        for _ in 0..6 {
            script.push(Err(ProviderError::Http {
                status: 500,
                body: "boom".into(),
            }));
        }
        let provider = FakeProvider::new(script);
        let agent = Agent::new(
            "A",
            "m/primary",
            vec!["m/fallback".into()],
            "sys",
            provider.clone(),
        );
        let err = agent.generate("q", 100, 0.7, None).await.unwrap_err();
        assert!(matches!(err, FusionError::AllModelsFailed { .. }));
        // 3 attempts per model * 2 models.
        assert_eq!(provider.calls.lock().unwrap().len(), 6);
    }

    #[tokio::test(start_paused = true)]
    async fn empty_choices_is_non_retryable() {
        let provider = FakeProvider::new(vec![
            Err(ProviderError::EmptyChoices {
                body: "{\"error\":\"x\"}".into(),
            }),
            Ok(FakeProvider::ok("fallback ok")),
        ]);
        let agent = Agent::new(
            "A",
            "m/primary",
            vec!["m/fallback".into()],
            "sys",
            provider.clone(),
        );
        let res = agent.generate("q", 100, 0.7, None).await.unwrap();
        assert_eq!(res.content, "fallback ok");
        // No retry on primary: straight to fallback.
        assert_eq!(provider.calls.lock().unwrap().len(), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn builds_system_and_user_messages() {
        let provider = FakeProvider::new(vec![Ok(FakeProvider::ok("ok"))]);
        let agent = Agent::new("A", "m/x", vec![], "you are A", provider.clone());
        agent.generate("hello", 50, 0.3, Some(7)).await.unwrap();
        let calls = provider.calls.lock().unwrap();
        let req = &calls[0];
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0], Message::system("you are A"));
        assert_eq!(req.messages[1], Message::user("hello"));
        assert_eq!(req.seed, Some(7));
        assert_eq!(req.max_tokens, 50);
    }
}
