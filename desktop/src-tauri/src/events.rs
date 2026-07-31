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

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Progress {
    /// "python" while the runtime downloads and extracts, "deps" during pip,
    /// "server" while waiting for the child to answer.
    pub phase: &'static str,
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
