//! End-to-end integration tests: a real `OpenRouterClient` driving a `Fusion`
//! debate against a `wiremock` server. Hermetic — no real key, no real network
//! (the base URL is pointed at the local mock).
//!
//! Retry/backoff timing is covered by the unit tests in `agent.rs` (with a
//! paused clock); here we focus on the wire round-trip: serialization, auth
//! header, request counts, fallback-model selection, and JSONL logging.

use std::sync::Arc;

use fusion::config::{AgentConfig, Config, SynthesizerConfig};
use fusion::fusion::Fusion;
use fusion::openrouter::OpenRouterClient;
use fusion::provider::ChatProvider;
use indexmap::IndexMap;
use serde_json::json;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// The mock endpoint path the client will POST to.
const ENDPOINT: &str = "/api/v1/chat/completions";

fn ok_completion(content: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "choices": [{ "message": { "role": "assistant", "content": content } }],
        "usage": { "prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7 }
    }))
}

fn provider_for(server: &MockServer) -> Arc<dyn ChatProvider> {
    let client = OpenRouterClient::new("test-key", IndexMap::new())
        .expect("client builds")
        .with_base_url(format!("{}{}", server.uri(), ENDPOINT));
    Arc::new(client)
}

fn two_agent_config(rounds: u32, log_file: std::path::PathBuf) -> Config {
    Config {
        api_key: Some("test-key".into()),
        agents: vec![
            AgentConfig {
                name: "Alice".into(),
                model: "m/alice".into(),
                role: Some("reasoning".into()),
                fallback_models: vec![],
            },
            AgentConfig {
                name: "Bob".into(),
                model: "m/bob".into(),
                role: Some("facts".into()),
                fallback_models: vec![],
            },
        ],
        synthesizer: SynthesizerConfig {
            name: "Alice".into(),
            model: "m/synth".into(),
            fallback_models: vec![],
        },
        rounds,
        max_tokens: 64,
        temperature: 0.7,
        seed: None,
        log_file: Some(log_file),
        extra_headers: IndexMap::new(),
    }
}

#[tokio::test]
async fn full_debate_round_trips_through_openrouter() {
    let server = MockServer::start().await;
    // Every completion returns a canned answer; assert the auth header too.
    Mock::given(method("POST"))
        .and(path(ENDPOINT))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ok_completion("the synthesized answer"))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("runs.jsonl");
    let cfg = two_agent_config(1, log.clone());
    let fusion = Fusion::from_config(&cfg, provider_for(&server)).unwrap();

    let (answer, meta) = fusion
        .debate("What is 2+2?", false, &mut |_| {})
        .await
        .expect("debate succeeds");

    assert_eq!(answer, "the synthesized answer");
    assert_eq!(meta.agents, vec!["Alice", "Bob"]);
    assert_eq!(meta.final_usage.total_tokens, 7);

    // 2 agents * (1 initial + 1 review) + 1 synthesis = 5 HTTP requests.
    let received = server.received_requests().await.unwrap();
    assert_eq!(received.len(), 5);

    // The JSONL log has exactly one row per step.
    let contents = std::fs::read_to_string(&log).unwrap();
    assert_eq!(contents.lines().count(), 5);
    for line in contents.lines() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        assert!(v["phase"].is_string());
        assert!(v["request"]["prompt"].is_string());
    }
}

#[tokio::test]
async fn request_body_serializes_expected_fields() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(ENDPOINT))
        .and(body_partial_json(
            json!({ "model": "m/solo", "max_tokens": 64 }),
        ))
        .respond_with(ok_completion("ok"))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let cfg = Config {
        api_key: Some("test-key".into()),
        agents: vec![AgentConfig {
            name: "Solo".into(),
            model: "m/solo".into(),
            role: None,
            fallback_models: vec![],
        }],
        synthesizer: SynthesizerConfig {
            name: "Solo".into(),
            model: "m/solo".into(),
            fallback_models: vec![],
        },
        rounds: 0,
        max_tokens: 64,
        temperature: 0.7,
        seed: None,
        log_file: Some(dir.path().join("r.jsonl")),
        extra_headers: IndexMap::new(),
    };
    let fusion = Fusion::from_config(&cfg, provider_for(&server)).unwrap();
    let (answer, _) = fusion.debate("hi", false, &mut |_| {}).await.unwrap();
    assert_eq!(answer, "ok");
    // initial + synthesis = 2 requests, both matched the body matcher.
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn non_retryable_primary_falls_back_to_secondary_model() {
    let server = MockServer::start().await;
    // Primary model 404s (non-retryable) → advance to the fallback model.
    Mock::given(method("POST"))
        .and(path(ENDPOINT))
        .and(body_partial_json(json!({ "model": "m/primary" })))
        .respond_with(ResponseTemplate::new(404).set_body_string("no such model"))
        .mount(&server)
        .await;
    // Fallback model and synthesizer succeed.
    Mock::given(method("POST"))
        .and(path(ENDPOINT))
        .and(body_partial_json(json!({ "model": "m/fallback" })))
        .respond_with(ok_completion("from fallback"))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(ENDPOINT))
        .and(body_partial_json(json!({ "model": "m/synth" })))
        .respond_with(ok_completion("final"))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let cfg = Config {
        api_key: Some("test-key".into()),
        agents: vec![AgentConfig {
            name: "Solo".into(),
            model: "m/primary".into(),
            role: None,
            fallback_models: vec!["m/fallback".into()],
        }],
        synthesizer: SynthesizerConfig {
            name: "Solo".into(),
            model: "m/synth".into(),
            fallback_models: vec![],
        },
        rounds: 0,
        max_tokens: 32,
        temperature: 0.7,
        seed: None,
        log_file: Some(dir.path().join("r.jsonl")),
        extra_headers: IndexMap::new(),
    };
    let fusion = Fusion::from_config(&cfg, provider_for(&server)).unwrap();
    let (answer, _) = fusion.debate("q", false, &mut |_| {}).await.unwrap();
    assert_eq!(answer, "final");
    // initial: 1 (primary 404) + 1 (fallback 200); synthesis: 1 = 3 requests.
    assert_eq!(server.received_requests().await.unwrap().len(), 3);
}
