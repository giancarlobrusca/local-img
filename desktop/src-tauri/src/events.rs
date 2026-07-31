//! The contract between the shell and its own first-run screens.
//!
//! Kept in one file so the event names and the field names the HTML reads are
//! visible together. Payloads are camelCase because they are read by
//! JavaScript, not by Rust.

use serde::Serialize;

/// Bootstrap progress. Two phases, each running 0..=100 in its own right.
pub const PROGRESS: &str = "shell://progress";
/// Something went wrong and the failure screen should take over.
pub const FAILED: &str = "shell://failed";
/// The uninstall finished and the removed screen should take over.
pub const REMOVED: &str = "shell://removed";

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Progress {
    /// "python" while the runtime downloads and extracts, "deps" during pip,
    /// "server" while waiting for the child to answer.
    pub phase: &'static str,
    /// 0..=100, or 255 meaning "this is subtext only — leave the bar alone".
    /// pip prints far more often than the bar moves; tying the two together
    /// would make it stutter.
    pub pct: u8,
    /// The line shown underneath the bar. pip's current output, mostly.
    pub detail: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Failure {
    /// One short line, in the user's terms.
    pub title: String,
    /// The real error, verbatim. Never a paraphrase.
    pub message: String,
    /// Everything a bug report needs, for the copy button.
    pub diagnostics: String,
    /// Whether retrying could plausibly work — network, yes; unsupported
    /// hardware, no.
    pub retryable: bool,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateInfo {
    pub version: String,
    pub url: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Removed {
    /// The total, already formatted: "23.4 GB". Composed from what Python
    /// reported and what the shell measured, never from either alone.
    pub freed: String,
    /// Absolute paths that could not be deleted. Empty on a clean run, and
    /// shown verbatim when it is not — a summary claiming 23 GB when it freed
    /// 19 is worse than no button at all.
    pub resisted: Vec<String>,
    /// The data directory, named on screen when something resisted so the user
    /// can go and look.
    pub data_dir: String,
    /// The one thing left to do, in words that match this platform.
    pub last_step: String,
    /// The label of the button that opens where that happens.
    pub open_label: String,
}
