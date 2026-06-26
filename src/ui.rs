//! Terminal progress UI for a running debate, driven by [`ProgressEvent`]s.
//!
//! NOTE: Stage-0 stub pinning the public API. The real `indicatif`-based
//! renderer (spinners per agent, phase headers, TTY degradation) is filled in
//! during Stage 3.

use crate::fusion::ProgressEvent;

/// Renders [`ProgressEvent`]s to the terminal using `indicatif`.
pub struct ProgressReporter {
    enabled: bool,
}

impl ProgressReporter {
    /// Create a reporter. When `enabled` is false (e.g. `--no-progress` or a
    /// non-TTY stdout) events are ignored so output stays clean for piping.
    pub fn new(enabled: bool) -> Self {
        ProgressReporter { enabled }
    }

    /// Handle a single progress event.
    pub fn handle(&mut self, _event: &ProgressEvent) {
        if !self.enabled {
            // no-op
        }
    }
}
