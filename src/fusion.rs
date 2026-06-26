//! The debate orchestrator.
//!
//! [`Fusion::debate`] runs the three faithful phases ported from the Python:
//! 1. **Initial** — every agent answers the query.
//! 2. **Review rounds** (N) — each agent critiques peers' latest *non-empty*
//!    answers (excluding its own) and emits a refined answer.
//! 3. **Synthesis** — the synthesizer merges the non-empty refined answers
//!    (falling back to all of them if none are non-empty) into one response.
//!
//! Agents run sequentially within each phase so the JSONL run-log ordering and
//! the provider call-count are deterministic (and therefore testable). A failed
//! agent contributes an empty string and is logged with its error, but does not
//! abort the run; only a failed *synthesizer* propagates an error.

use crate::agent::Agent;
use crate::config::{self, Config};
use crate::error::Result;
use crate::logging::JsonlLogger;
use crate::prompts;
use crate::provider::{ChatProvider, Usage};
use indexmap::IndexMap;
use serde_json::json;
use std::sync::Arc;

/// Progress events emitted during a debate, consumed by the UI layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgressEvent {
    /// A free-form status line (e.g. "Starting debate...").
    Status(String),
    /// A new phase began (e.g. "Initial", "Review round 1/3", "Synthesis").
    PhaseStart { phase: String },
    /// An agent's generation started within a phase.
    AgentStart { phase: String, agent: String },
    /// An agent's generation finished. `ok` is false on failure/empty output.
    AgentDone {
        phase: String,
        agent: String,
        chars: usize,
        ok: bool,
    },
    /// The debate finished.
    Done,
}

/// Metadata describing a completed debate run.
#[derive(Debug, Clone)]
pub struct RunMeta {
    pub query: String,
    pub rounds: u32,
    pub agents: Vec<String>,
    pub final_answer: String,
    pub final_usage: Usage,
}

/// The debate engine: a roster of agents plus a synthesizer.
pub struct Fusion {
    agents: Vec<Agent>,
    synthesizer: Agent,
    rounds: u32,
    max_tokens: u32,
    temperature: f32,
    seed: Option<u64>,
    logger: JsonlLogger,
}

impl Fusion {
    /// Build a [`Fusion`] from a validated [`Config`] and a shared provider.
    pub fn from_config(cfg: &Config, provider: Arc<dyn ChatProvider>) -> Result<Self> {
        cfg.validate()?;

        let agents = cfg
            .agents
            .iter()
            .map(|a| {
                Agent::new(
                    a.name.clone(),
                    a.model.clone(),
                    a.fallback_models.clone(),
                    prompts::build_system_prompt(&a.name, a.role.as_deref()),
                    provider.clone(),
                )
            })
            .collect();

        let synthesizer = Agent::new(
            format!("Synthesizer({})", cfg.synthesizer.name),
            cfg.synthesizer.model.clone(),
            cfg.synthesizer.fallback_models.clone(),
            prompts::synthesizer_system_prompt(),
            provider.clone(),
        );

        let log_path = cfg
            .log_file
            .clone()
            .or_else(config::default_log_path)
            .unwrap_or_else(|| std::path::PathBuf::from("runs.jsonl"));

        Ok(Fusion {
            agents,
            synthesizer,
            rounds: cfg.rounds,
            max_tokens: cfg.max_tokens,
            temperature: cfg.temperature,
            seed: cfg.seed,
            logger: JsonlLogger::new(log_path),
        })
    }

    /// The JSONL run-log path this engine writes to.
    pub fn log_path(&self) -> &std::path::Path {
        self.logger.path()
    }

    /// Run the full debate, emitting progress to `on_event`.
    pub async fn debate(
        &self,
        query: &str,
        paper_mode: bool,
        on_event: &mut dyn FnMut(ProgressEvent),
    ) -> Result<(String, RunMeta)> {
        on_event(ProgressEvent::Status("Starting debate...".into()));

        // ---- Phase 1: initial generation ------------------------------------
        let phase = "Initial".to_string();
        on_event(ProgressEvent::PhaseStart {
            phase: phase.clone(),
        });
        let mut agent_latest: IndexMap<String, String> = IndexMap::new();
        for agent in &self.agents {
            on_event(ProgressEvent::AgentStart {
                phase: phase.clone(),
                agent: agent.name.clone(),
            });
            let prompt = prompts::build_initial_prompt(query, paper_mode);
            let (content, response_value) = self.run_step(agent, &prompt).await;
            self.logger
                .log_step("initial", &agent.name, &agent.model, &prompt, &response_value);
            on_event(ProgressEvent::AgentDone {
                phase: phase.clone(),
                agent: agent.name.clone(),
                chars: content.len(),
                ok: !content.trim().is_empty(),
            });
            agent_latest.insert(agent.name.clone(), content);
        }

        // ---- Phase 2: review rounds -----------------------------------------
        for r in 1..=self.rounds {
            let phase = format!("Review round {r}/{}", self.rounds);
            on_event(ProgressEvent::PhaseStart {
                phase: phase.clone(),
            });
            let mut new_latest: IndexMap<String, String> = IndexMap::new();
            for agent in &self.agents {
                on_event(ProgressEvent::AgentStart {
                    phase: phase.clone(),
                    agent: agent.name.clone(),
                });
                // Peers: every other agent's latest *non-empty* answer.
                let others: IndexMap<String, String> = agent_latest
                    .iter()
                    .filter(|(name, content)| {
                        name.as_str() != agent.name && !content.trim().is_empty()
                    })
                    .map(|(name, content)| (name.clone(), content.clone()))
                    .collect();
                let self_prev = agent_latest.get(&agent.name).cloned().unwrap_or_default();
                let prompt =
                    prompts::build_review_prompt(query, &self_prev, &others, paper_mode);
                let (raw_content, response_value) = self.run_step(agent, &prompt).await;
                let refined = prompts::extract_refined(&raw_content);
                self.logger.log_step(
                    &format!("review_{r}"),
                    &agent.name,
                    &agent.model,
                    &prompt,
                    &response_value,
                );
                on_event(ProgressEvent::AgentDone {
                    phase: phase.clone(),
                    agent: agent.name.clone(),
                    chars: refined.len(),
                    ok: !refined.trim().is_empty(),
                });
                new_latest.insert(agent.name.clone(), refined);
            }
            agent_latest = new_latest;
        }

        // ---- Phase 3: synthesis ---------------------------------------------
        let phase = "Synthesis".to_string();
        on_event(ProgressEvent::PhaseStart {
            phase: phase.clone(),
        });
        on_event(ProgressEvent::AgentStart {
            phase: phase.clone(),
            agent: self.synthesizer.name.clone(),
        });
        // Synthesize over genuine, non-empty contributions; if every agent
        // degraded, fall back to all of them so the run still produces a result.
        let mut synth_inputs: IndexMap<String, String> = agent_latest
            .iter()
            .filter(|(_, content)| !content.trim().is_empty())
            .map(|(name, content)| (name.clone(), content.clone()))
            .collect();
        if synth_inputs.is_empty() {
            synth_inputs = agent_latest.clone();
        }
        let synth_prompt = prompts::build_synthesis_prompt(query, &synth_inputs, paper_mode);
        // A failed synthesizer is fatal — it is the final answer.
        let synth_res = self
            .synthesizer
            .generate(&synth_prompt, self.max_tokens, self.temperature, self.seed)
            .await?;
        self.logger.log_step(
            "synthesis",
            &self.synthesizer.name,
            &self.synthesizer.model,
            &synth_prompt,
            &synth_res.raw,
        );
        let final_answer = synth_res.content;
        on_event(ProgressEvent::AgentDone {
            phase,
            agent: self.synthesizer.name.clone(),
            chars: final_answer.len(),
            ok: !final_answer.trim().is_empty(),
        });
        on_event(ProgressEvent::Done);

        let meta = RunMeta {
            query: query.to_string(),
            rounds: self.rounds,
            agents: self.agents.iter().map(|a| a.name.clone()).collect(),
            final_answer: final_answer.clone(),
            final_usage: synth_res.usage,
        };
        Ok((final_answer, meta))
    }

    /// Run one agent generation, returning `(content, response_json)`. A failed
    /// generation yields empty content and an error-shaped JSON value for the
    /// run log (it does not abort the debate).
    async fn run_step(&self, agent: &Agent, prompt: &str) -> (String, serde_json::Value) {
        match agent
            .generate(prompt, self.max_tokens, self.temperature, self.seed)
            .await
        {
            Ok(resp) => (resp.content, resp.raw),
            Err(e) => {
                tracing::warn!("{} failed to generate: {e}", agent.name);
                (String::new(), json!({ "error": e.to_string(), "content": "" }))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AgentConfig, SynthesizerConfig};
    use crate::error::{FusionError, ProviderError};
    use crate::provider::{ChatRequest, ChatResponse};
    use async_trait::async_trait;
    use std::sync::Mutex;

    /// Provider that echoes a deterministic answer derived from the model id,
    /// and counts calls. Lets us assert call counts and final synthesis.
    struct CountingProvider {
        calls: Mutex<Vec<ChatRequest>>,
    }
    impl CountingProvider {
        fn new() -> Arc<Self> {
            Arc::new(CountingProvider {
                calls: Mutex::new(Vec::new()),
            })
        }
    }
    #[async_trait]
    impl ChatProvider for CountingProvider {
        async fn chat(&self, req: &ChatRequest) -> std::result::Result<ChatResponse, ProviderError> {
            self.calls.lock().unwrap().push(req.clone());
            let content = format!("answer from {}", req.model);
            Ok(ChatResponse {
                content,
                usage: Usage::default(),
                raw: json!({"choices":[{"message":{"content":"x"}}]}),
            })
        }
    }

    fn test_config(rounds: u32, log_file: std::path::PathBuf) -> Config {
        Config {
            api_key: Some("k".into()),
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
            max_tokens: 100,
            temperature: 0.7,
            seed: None,
            log_file: Some(log_file),
            extra_headers: Default::default(),
        }
    }

    #[tokio::test]
    async fn debate_call_count_is_agents_plus_reviews_plus_one() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = test_config(2, dir.path().join("runs.jsonl"));
        let provider = CountingProvider::new();
        let fusion = Fusion::from_config(&cfg, provider.clone()).unwrap();
        let mut events = Vec::new();
        let (answer, meta) = fusion
            .debate("Q", false, &mut |e| events.push(e))
            .await
            .unwrap();
        // 2 agents * (1 initial + 2 reviews) + 1 synthesis = 7 provider calls.
        assert_eq!(provider.calls.lock().unwrap().len(), 7);
        assert_eq!(answer, "answer from m/synth");
        assert_eq!(meta.agents, vec!["Alice", "Bob"]);
        assert!(events.contains(&ProgressEvent::Done));
    }

    #[tokio::test]
    async fn debate_writes_one_jsonl_row_per_step() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("runs.jsonl");
        let cfg = test_config(1, log.clone());
        let provider = CountingProvider::new();
        let fusion = Fusion::from_config(&cfg, provider).unwrap();
        fusion.debate("Q", false, &mut |_| {}).await.unwrap();
        let contents = std::fs::read_to_string(&log).unwrap();
        // 2 initial + 2 review_1 + 1 synthesis = 5 rows.
        assert_eq!(contents.lines().count(), 5);
    }

    #[tokio::test]
    async fn duplicate_agent_names_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = test_config(1, dir.path().join("r.jsonl"));
        cfg.agents[1].name = "Alice".into();
        let provider = CountingProvider::new();
        match Fusion::from_config(&cfg, provider) {
            Err(FusionError::DuplicateAgentNames(_)) => {}
            other => panic!("expected DuplicateAgentNames, got {:?}", other.err()),
        }
    }

    #[tokio::test]
    async fn failed_synthesizer_propagates() {
        // Provider that always fails non-retryably.
        struct DeadProvider;
        #[async_trait]
        impl ChatProvider for DeadProvider {
            async fn chat(
                &self,
                _req: &ChatRequest,
            ) -> std::result::Result<ChatResponse, ProviderError> {
                Err(ProviderError::Http {
                    status: 401,
                    body: "no key".into(),
                })
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let cfg = test_config(1, dir.path().join("r.jsonl"));
        let fusion = Fusion::from_config(&cfg, Arc::new(DeadProvider)).unwrap();
        let err = fusion.debate("Q", false, &mut |_| {}).await.unwrap_err();
        assert!(matches!(err, FusionError::AllModelsFailed { .. }));
    }
}
