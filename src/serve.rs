//! OpenAI-compatible HTTP server.
//!
//! Exposes the debate engine behind a subset of the OpenAI Chat Completions API
//! so existing tools — Codex CLI, opencode, anything that speaks
//! `POST /v1/chat/completions` against a custom base URL — can use FUSION as if
//! it were a single model. A request's `messages` are flattened into one query,
//! a full debate runs, and the synthesized answer comes back as the assistant
//! message.
//!
//! Notes / deliberate limitations:
//! - The request `model` field is **ignored** for routing: FUSION always runs the
//!   roster from its own config. It is echoed back in the response for clients
//!   that assert on it.
//! - `stream: true` is supported, but the answer is produced all at once at the
//!   end of the debate, so it is delivered as a single content chunk followed by
//!   `[DONE]` rather than token-by-token.
//! - The endpoint is unauthenticated; bind it to localhost (the default) or put
//!   it behind your own proxy. Inbound `Authorization` headers are ignored — the
//!   server authenticates to OpenRouter with its own configured key.

use crate::fusion::Fusion;
use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Shared handler state.
#[derive(Clone)]
struct AppState {
    fusion: Arc<Fusion>,
    paper_mode: bool,
}

/// A chat message as sent by an OpenAI-compatible client. `content` may be a
/// plain string, an array of content parts (`{"type":"text","text":...}`), or
/// `null` (e.g. an assistant turn carrying only tool calls).
#[derive(Debug, Deserialize)]
struct IncomingMessage {
    #[serde(default)]
    role: String,
    #[serde(default)]
    content: Option<Value>,
}

/// The subset of the Chat Completions request body we read. Unknown fields
/// (`temperature`, `tools`, `top_p`, ...) are accepted and ignored.
#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    messages: Vec<IncomingMessage>,
    #[serde(default)]
    stream: bool,
}

/// Start the server and block until it shuts down (Ctrl-C).
pub async fn run_server(
    fusion: Arc<Fusion>,
    host: &str,
    port: u16,
    paper_mode: bool,
) -> anyhow::Result<()> {
    let state = AppState { fusion, paper_mode };
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/health", get(|| async { "ok" }))
        .with_state(state);

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| anyhow::anyhow!("failed to bind {addr}: {e}"))?;

    eprintln!("FUSION OpenAI-compatible server listening on http://{addr}");
    eprintln!("  POST /v1/chat/completions   (the `model` field is ignored; the configured roster always runs)");
    eprintln!("  GET  /v1/models");
    eprintln!("Point a client at  http://{addr}/v1  with any API key. Ctrl-C to stop.");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| anyhow::anyhow!("server error: {e}"))?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    eprintln!("\nshutting down");
}

/// `GET /v1/models` — advertise the single virtual "fusion" model.
async fn list_models() -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": "fusion",
            "object": "model",
            "created": unix_secs(),
            "owned_by": "fusion",
        }],
    }))
}

/// `POST /v1/chat/completions` — run a debate and return the synthesized answer.
async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    if req.messages.is_empty() {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "`messages` must not be empty",
        );
    }

    let query = messages_to_query(&req.messages);
    if query.trim().is_empty() {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "no usable text content found in `messages`",
        );
    }

    let model_echo = req.model.clone().unwrap_or_else(|| "fusion".to_string());

    let mut sink = |_event| {};
    match state
        .fusion
        .debate(&query, state.paper_mode, &mut sink)
        .await
    {
        Ok((answer, meta)) => {
            let created = unix_secs();
            let id = completion_id();
            if req.stream {
                stream_response(&id, created, &model_echo, &answer)
            } else {
                json_response(&id, created, &model_echo, &answer, &meta)
            }
        }
        Err(e) => api_error(
            StatusCode::BAD_GATEWAY,
            "upstream_error",
            &format!("debate failed: {e}"),
        ),
    }
}

/// Flatten an OpenAI message list into a single debate query.
///
/// A lone user turn (optionally preceded by system messages) is passed through
/// mostly verbatim — the common single-shot case. A multi-turn conversation is
/// rendered as a simple role-tagged transcript so the debate sees the history.
fn messages_to_query(messages: &[IncomingMessage]) -> String {
    let rendered: Vec<(&str, String)> = messages
        .iter()
        .map(|m| (m.role.as_str(), content_to_text(m.content.as_ref())))
        .filter(|(_, text)| !text.trim().is_empty())
        .collect();

    let non_system: Vec<&(&str, String)> = rendered
        .iter()
        .filter(|(role, _)| *role != "system")
        .collect();

    // Single user turn: prepend any system context, then the user text.
    if non_system.len() == 1 && non_system[0].0 == "user" {
        let mut out = String::new();
        for (role, text) in &rendered {
            if *role == "system" {
                out.push_str(text);
                out.push_str("\n\n");
            }
        }
        out.push_str(&non_system[0].1);
        return out.trim().to_string();
    }

    // Otherwise, a role-tagged transcript.
    let mut out = String::new();
    for (role, text) in &rendered {
        out.push_str(&format!("[{role}]\n{text}\n\n"));
    }
    out.trim().to_string()
}

/// Coerce an OpenAI `content` field (string | array of parts | null) to text.
fn content_to_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// Build a non-streaming `chat.completion` response.
fn json_response(
    id: &str,
    created: u64,
    model: &str,
    answer: &str,
    meta: &crate::fusion::RunMeta,
) -> Response {
    let body = json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": answer },
            "finish_reason": "stop",
        }],
        // Best-effort: only the synthesizer's token usage is known here.
        "usage": {
            "prompt_tokens": meta.final_usage.prompt_tokens,
            "completion_tokens": meta.final_usage.completion_tokens,
            "total_tokens": meta.final_usage.total_tokens,
        },
    });
    (StatusCode::OK, Json(body)).into_response()
}

/// Build a streaming (SSE) response. The whole answer is delivered as one content
/// chunk, since the debate yields it all at once, then `[DONE]`.
fn stream_response(id: &str, created: u64, model: &str, answer: &str) -> Response {
    let chunk = |delta: Value, finish: Value| {
        let obj = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
        });
        format!("data: {obj}\n\n")
    };

    let mut body = String::new();
    body.push_str(&chunk(json!({ "role": "assistant" }), Value::Null));
    body.push_str(&chunk(json!({ "content": answer }), Value::Null));
    body.push_str(&chunk(json!({}), json!("stop")));
    body.push_str("data: [DONE]\n\n");

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/event-stream")],
        body,
    )
        .into_response()
}

/// An OpenAI-shaped error response.
fn api_error(status: StatusCode, err_type: &str, message: &str) -> Response {
    let body = json!({
        "error": { "message": message, "type": err_type },
    });
    (status, Json(body)).into_response()
}

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn completion_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("chatcmpl-{nanos:x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AgentConfig, Config, SynthesizerConfig};
    use crate::error::ProviderError;
    use crate::provider::{ChatProvider, ChatRequest, ChatResponse, Usage};
    use async_trait::async_trait;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt; // for `oneshot`

    fn msg(role: &str, content: Value) -> IncomingMessage {
        IncomingMessage {
            role: role.into(),
            content: Some(content),
        }
    }

    #[test]
    fn single_user_turn_passes_through() {
        let q = messages_to_query(&[msg("user", json!("hello world"))]);
        assert_eq!(q, "hello world");
    }

    #[test]
    fn system_then_user_is_prepended() {
        let q = messages_to_query(&[msg("system", json!("be terse")), msg("user", json!("hi"))]);
        assert_eq!(q, "be terse\n\nhi");
    }

    #[test]
    fn multi_turn_renders_transcript() {
        let q = messages_to_query(&[
            msg("user", json!("a")),
            msg("assistant", json!("b")),
            msg("user", json!("c")),
        ]);
        assert_eq!(q, "[user]\na\n\n[assistant]\nb\n\n[user]\nc");
    }

    #[test]
    fn content_parts_array_is_flattened() {
        let q = messages_to_query(&[msg(
            "user",
            json!([
                {"type": "text", "text": "part1 "},
                {"type": "text", "text": "part2"}
            ]),
        )]);
        assert_eq!(q, "part1 part2");
    }

    #[test]
    fn null_content_is_ignored() {
        let q = messages_to_query(&[
            IncomingMessage {
                role: "assistant".into(),
                content: None,
            },
            msg("user", json!("real question")),
        ]);
        assert_eq!(q, "real question");
    }

    /// Provider that returns a deterministic answer keyed by model id.
    struct FakeProvider;
    #[async_trait]
    impl ChatProvider for FakeProvider {
        async fn chat(
            &self,
            req: &ChatRequest,
        ) -> std::result::Result<ChatResponse, ProviderError> {
            Ok(ChatResponse {
                content: format!("answer from {}", req.model),
                usage: Usage {
                    prompt_tokens: 3,
                    completion_tokens: 5,
                    total_tokens: 8,
                },
                raw: json!({}),
            })
        }
    }

    fn test_app() -> Router {
        let dir = tempfile::tempdir().unwrap();
        let cfg = Config {
            api_key: Some("k".into()),
            agents: vec![
                AgentConfig {
                    name: "A".into(),
                    model: "m/a".into(),
                    role: None,
                    fallback_models: vec![],
                },
                AgentConfig {
                    name: "B".into(),
                    model: "m/b".into(),
                    role: None,
                    fallback_models: vec![],
                },
            ],
            synthesizer: SynthesizerConfig {
                name: "S".into(),
                model: "m/synth".into(),
                fallback_models: vec![],
            },
            rounds: 1,
            max_tokens: 64,
            temperature: 0.7,
            seed: None,
            // Keep the run log inside the temp dir for the test's lifetime.
            log_file: Some(dir.keep().join("runs.jsonl")),
            extra_headers: Default::default(),
        };
        let fusion = Fusion::from_config(&cfg, Arc::new(FakeProvider)).unwrap();
        Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/models", get(list_models))
            .with_state(AppState {
                fusion: Arc::new(fusion),
                paper_mode: false,
            })
    }

    async fn body_json(resp: Response) -> Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn chat_completions_returns_synthesized_answer() {
        let app = test_app();
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Explain quicksort"}]
                })
                .to_string(),
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["object"], "chat.completion");
        // The request model is echoed back, not used for routing.
        assert_eq!(v["model"], "gpt-4");
        assert_eq!(v["choices"][0]["message"]["content"], "answer from m/synth");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["total_tokens"], 8);
    }

    #[tokio::test]
    async fn empty_messages_is_a_400() {
        let app = test_app();
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(json!({ "messages": [] }).to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = body_json(resp).await;
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn streaming_emits_sse_chunks_and_done() {
        let app = test_app();
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "stream": true,
                    "messages": [{"role": "user", "content": "hi"}]
                })
                .to_string(),
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        assert!(ct.starts_with("text/event-stream"));
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains("\"content\":\"answer from m/synth\""));
        assert!(text.trim().ends_with("data: [DONE]"));
    }

    #[tokio::test]
    async fn models_endpoint_lists_fusion() {
        let app = test_app();
        let req = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["data"][0]["id"], "fusion");
    }
}
