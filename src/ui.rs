//! Terminal progress UI for a running debate, driven by [`ProgressEvent`]s.
//!
//! Renders a clean live display to stderr using `indicatif`: a bold phase
//! header per [`ProgressEvent::PhaseStart`], a steady-tick spinner per
//! [`ProgressEvent::AgentStart`], and a checkmark/cross when an agent finishes.
//! When disabled (e.g. `--no-progress` or a non-TTY stdout) every method is a
//! no-op so piped output stays clean.

use std::time::Duration;

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

use crate::fusion::ProgressEvent;

/// Steady-tick interval for the active agent spinner.
const TICK: Duration = Duration::from_millis(100);

/// Renders [`ProgressEvent`]s to the terminal using `indicatif`.
pub struct ProgressReporter {
    enabled: bool,
    /// The spinner for the currently running agent, if any. Agents run
    /// sequentially, so a single reusable handle is sufficient.
    spinner: Option<ProgressBar>,
    /// When set, spinners draw to a hidden target (used by tests).
    hidden: bool,
}

impl ProgressReporter {
    /// Create a reporter. When `enabled` is false (e.g. `--no-progress` or a
    /// non-TTY stdout) events are ignored so output stays clean for piping.
    /// The TTY/flag decision is made by the CLI; this constructor only takes
    /// the resulting bool.
    pub fn new(enabled: bool) -> Self {
        ProgressReporter {
            enabled,
            spinner: None,
            hidden: false,
        }
    }

    /// Like [`ProgressReporter::new`], but routes all spinner output to a hidden
    /// draw target. Used by tests so they never touch a real terminal.
    #[cfg(test)]
    fn new_hidden(enabled: bool) -> Self {
        ProgressReporter {
            enabled,
            spinner: None,
            hidden: true,
        }
    }

    /// Build the spinner style, falling back to a bare default if the (static,
    /// valid) template ever fails to parse.
    fn spinner_style() -> ProgressStyle {
        ProgressStyle::with_template("{spinner:.cyan} {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_spinner())
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", "✓"])
    }

    /// Create a fresh spinner, honoring the hidden draw target in tests.
    fn make_spinner(&self) -> ProgressBar {
        let pb = ProgressBar::new_spinner();
        if self.hidden {
            pb.set_draw_target(ProgressDrawTarget::hidden());
        }
        pb.set_style(Self::spinner_style());
        pb
    }

    /// Print a line without corrupting an active spinner. Routes to the
    /// spinner's `println` when one is live (so it redraws cleanly), otherwise
    /// straight to stderr.
    fn emit_line(&self, line: &str) {
        match &self.spinner {
            Some(pb) => pb.println(line),
            None => eprintln!("{line}"),
        }
    }

    /// Finish and drop any active spinner without leaving residue.
    fn clear_spinner(&mut self) {
        if let Some(pb) = self.spinner.take() {
            pb.finish_and_clear();
        }
    }

    /// Handle a single progress event.
    pub fn handle(&mut self, event: &ProgressEvent) {
        if !self.enabled {
            return;
        }

        match event {
            ProgressEvent::Status(msg) => {
                self.emit_line(msg);
            }
            ProgressEvent::PhaseStart { phase } => {
                // A new phase: retire any leftover spinner, then a header.
                self.clear_spinner();
                self.emit_line(&format!("▶ {phase}"));
            }
            ProgressEvent::AgentStart { phase: _, agent } => {
                // Sequential agents: replace any prior spinner.
                self.clear_spinner();
                let pb = self.make_spinner();
                pb.set_message(format!("{agent} …"));
                pb.enable_steady_tick(TICK);
                self.spinner = Some(pb);
            }
            ProgressEvent::AgentDone {
                phase: _,
                agent,
                chars,
                ok,
            } => {
                if let Some(pb) = self.spinner.take() {
                    let msg = if *ok {
                        format!("✓ {agent} ({chars} chars)")
                    } else {
                        format!("✗ {agent} (no output)")
                    };
                    pb.finish_with_message(msg);
                } else {
                    // No spinner was running; still report the outcome.
                    let msg = if *ok {
                        format!("✓ {agent} ({chars} chars)")
                    } else {
                        format!("✗ {agent} (no output)")
                    };
                    self.emit_line(&msg);
                }
            }
            ProgressEvent::Done => {
                self.clear_spinner();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_sequence() -> Vec<ProgressEvent> {
        vec![
            ProgressEvent::Status("Starting debate...".into()),
            ProgressEvent::PhaseStart {
                phase: "Initial".into(),
            },
            ProgressEvent::AgentStart {
                phase: "Initial".into(),
                agent: "alpha".into(),
            },
            ProgressEvent::AgentDone {
                phase: "Initial".into(),
                agent: "alpha".into(),
                chars: 1234,
                ok: true,
            },
            ProgressEvent::AgentStart {
                phase: "Initial".into(),
                agent: "beta".into(),
            },
            ProgressEvent::AgentDone {
                phase: "Initial".into(),
                agent: "beta".into(),
                chars: 0,
                ok: false,
            },
            ProgressEvent::Done,
        ]
    }

    #[test]
    fn disabled_reporter_is_a_noop_on_every_variant() {
        let mut r = ProgressReporter::new(false);
        for ev in full_sequence() {
            r.handle(&ev);
        }
        // Also exercise the standalone Status/AgentDone-without-spinner paths.
        r.handle(&ProgressEvent::Status("x".into()));
        assert!(r.spinner.is_none());
    }

    #[test]
    fn enabled_reporter_handles_full_sequence_without_panicking() {
        let mut r = ProgressReporter::new_hidden(true);
        for ev in full_sequence() {
            r.handle(&ev);
        }
        // After Done, no spinner should remain active.
        assert!(r.spinner.is_none());
    }

    #[test]
    fn agent_done_without_active_spinner_does_not_panic() {
        let mut r = ProgressReporter::new_hidden(true);
        r.handle(&ProgressEvent::AgentDone {
            phase: "Synthesis".into(),
            agent: "lone".into(),
            chars: 5,
            ok: true,
        });
        assert!(r.spinner.is_none());
    }
}
