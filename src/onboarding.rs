//! Interactive onboarding wizard.
//!
//! NOTE: Stage-0 stub pinning the public API. The real `dialoguer`-based wizard
//! (key prompt, live key validation, roster/rounds customization, config write)
//! is filled in during Stage 3.

use crate::config::Config;
use crate::error::Result;
use std::path::Path;

/// Run the interactive onboarding wizard, persisting the result to `save_path`
/// and returning the populated [`Config`] for immediate use.
pub async fn run_onboarding(_save_path: &Path) -> Result<Config> {
    unimplemented!("implemented in Stage 3")
}
