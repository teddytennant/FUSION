//! Best-effort JSONL run logging.
//!
//! Each debate step appends one JSON object per line, matching the original
//! Python schema: `{phase, agent, model, request: {prompt}, response}`. Writes
//! are best-effort — a logging failure must never abort a debate, so errors are
//! traced and swallowed.

use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Appends structured rows to a JSONL file.
#[derive(Debug, Clone)]
pub struct JsonlLogger {
    path: PathBuf,
}

impl JsonlLogger {
    /// Create a logger that appends to `path`. Parent directories are created
    /// lazily on first write.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        JsonlLogger { path: path.into() }
    }

    /// The path rows are appended to.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one row. Failures are logged via `tracing` and otherwise ignored.
    pub fn log(&self, row: &Value) {
        if let Err(e) = self.try_log(row) {
            tracing::error!("failed to write run log to {}: {e}", self.path.display());
        }
    }

    /// Convenience helper to build and append a standard debate-step row.
    pub fn log_step(&self, phase: &str, agent: &str, model: &str, prompt: &str, response: &Value) {
        let row = json!({
            "phase": phase,
            "agent": agent,
            "model": model,
            "request": { "prompt": prompt },
            "response": response,
        });
        self.log(&row);
    }

    fn try_log(&self, row: &Value) -> std::io::Result<()> {
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        let line = serde_json::to_string(row).unwrap_or_else(|_| "{}".to_string());
        f.write_all(line.as_bytes())?;
        f.write_all(b"\n")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_one_json_object_per_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub").join("runs.jsonl");
        let logger = JsonlLogger::new(&path);
        logger.log_step(
            "initial",
            "Alice",
            "m/x",
            "prompt-a",
            &json!({"content": "hi"}),
        );
        logger.log_step(
            "synthesis",
            "Synth",
            "m/y",
            "prompt-b",
            &json!({"content": "bye"}),
        );

        let contents = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2);
        for line in lines {
            let v: Value = serde_json::from_str(line).unwrap();
            assert!(v.get("phase").is_some());
            assert!(v["request"]["prompt"].is_string());
        }
    }
}
