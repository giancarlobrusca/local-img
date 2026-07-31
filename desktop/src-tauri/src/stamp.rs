//! `runtime.json` — the record of what was installed, and the only thing that
//! decides whether a launch bootstraps.
//!
//! Written **last** and atomically. An install interrupted anywhere before
//! that leaves no stamp, so the next launch redoes it instead of starting a
//! half-built environment that fails somewhere confusing.

use crate::layout::Layout;
use crate::plan::RuntimePlan;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stamp {
    pub python_version: String,
    pub triple: String,
    pub torch_index: String,
    pub requirements_sha256: String,
    /// Recorded for a bug report to read. Deliberately not part of the
    /// identity — comparing it would reinstall on every launch.
    pub written_at: String,
}

impl Stamp {
    /// What a correct installation of `plan` against these requirements looks
    /// like. Compared against what is on disk to decide whether to bootstrap.
    pub fn expected(plan: &RuntimePlan, requirements: &[u8]) -> Self {
        Self {
            python_version: crate::plan::PYTHON_VERSION.to_string(),
            triple: plan.triple.to_string(),
            torch_index: plan.index_label().to_string(),
            requirements_sha256: sha256_hex(requirements),
            written_at: now_iso8601(),
        }
    }

    pub fn matches(&self, other: &Self) -> bool {
        self.python_version == other.python_version
            && self.triple == other.triple
            && self.torch_index == other.torch_index
            && self.requirements_sha256 == other.requirements_sha256
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn now_iso8601() -> String {
    // Seconds since the epoch, formatted by hand. Pulling a date crate in for
    // one diagnostic field is not worth the dependency; what matters is that
    // two stamps written at different times are distinguishable in a log.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("epoch:{secs}")
}

pub fn read(path: &Path) -> Option<Stamp> {
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

/// Write via a temp file and a rename, so a crash mid-write cannot leave a
/// truncated stamp that parses as a different installation.
pub fn write_atomic(path: &Path, stamp: &Stamp) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temp = path.with_extension("json.tmp");
    {
        let mut file = std::fs::File::create(&temp)?;
        file.write_all(serde_json::to_string_pretty(stamp)?.as_bytes())?;
        // Rename is atomic; the bytes reaching the platter first is not.
        file.sync_all()?;
    }
    // Windows will not rename onto an existing file.
    let _ = std::fs::remove_file(path);
    std::fs::rename(&temp, path)
}

/// Whether this launch has to install the runtime.
///
/// Two independent reasons, and both have to be checked: the stamp can be
/// missing, stale, or describe a different plan; and the interpreter it claims
/// to describe can simply not be there.
pub fn needs_bootstrap(layout: &Layout, expected: &Stamp) -> bool {
    if !layout.interpreter.is_file() {
        return true;
    }
    match read(&layout.stamp_path) {
        Some(found) => !found.matches(expected),
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::plan_for;

    fn a_plan() -> crate::plan::RuntimePlan {
        plan_for("linux", "x86_64", false).unwrap()
    }

    fn a_stamp() -> Stamp {
        Stamp::expected(&a_plan(), b"torch>=2.4\n")
    }

    #[test]
    fn sha256_matches_the_reference_value() {
        // The empty-string digest, so a broken hasher cannot pass by accident.
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn a_stamp_round_trips_through_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("runtime.json");
        let stamp = a_stamp();
        write_atomic(&path, &stamp).unwrap();
        assert_eq!(read(&path).unwrap(), stamp);
    }

    #[test]
    fn writing_leaves_no_temp_file_behind() {
        let dir = tempfile::tempdir().unwrap();
        write_atomic(&dir.path().join("runtime.json"), &a_stamp()).unwrap();
        let names: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["runtime.json".to_string()]);
    }

    #[test]
    fn a_missing_or_corrupt_stamp_reads_as_none() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read(&dir.path().join("absent.json")).is_none());
        let path = dir.path().join("corrupt.json");
        std::fs::write(&path, b"{not json").unwrap();
        assert!(read(&path).is_none());
        std::fs::write(&path, br#"{"python_version": "3.12.13"}"#).unwrap();
        assert!(read(&path).is_none(), "a partial stamp is not a stamp");
    }

    #[test]
    fn a_matching_stamp_and_a_present_interpreter_skip_the_bootstrap() {
        let dir = tempfile::tempdir().unwrap();
        let layout = crate::layout::Layout::new(dir.path());
        std::fs::create_dir_all(layout.interpreter.parent().unwrap()).unwrap();
        std::fs::write(&layout.interpreter, b"#!/bin/sh\n").unwrap();
        write_atomic(&layout.stamp_path, &a_stamp()).unwrap();
        assert!(!needs_bootstrap(&layout, &a_stamp()));
    }

    #[test]
    fn each_changed_field_forces_a_redo() {
        let dir = tempfile::tempdir().unwrap();
        let layout = crate::layout::Layout::new(dir.path());
        std::fs::create_dir_all(layout.interpreter.parent().unwrap()).unwrap();
        std::fs::write(&layout.interpreter, b"#!/bin/sh\n").unwrap();
        write_atomic(&layout.stamp_path, &a_stamp()).unwrap();

        let mut newer_python = a_stamp();
        newer_python.python_version = "3.13.0".into();
        assert!(needs_bootstrap(&layout, &newer_python), "python version");

        let mut other_platform = a_stamp();
        other_platform.triple = "x86_64-pc-windows-msvc".into();
        assert!(needs_bootstrap(&layout, &other_platform), "triple");

        let mut cuda_now = a_stamp();
        cuda_now.torch_index = crate::plan::CUDA_INDEX.into();
        assert!(needs_bootstrap(&layout, &cuda_now), "torch index");

        let changed_requirements = Stamp::expected(&a_plan(), b"torch>=2.9\n");
        assert!(needs_bootstrap(&layout, &changed_requirements), "requirements");
    }

    #[test]
    fn the_timestamp_is_not_part_of_the_identity() {
        // Otherwise every launch would reinstall.
        let mut later = a_stamp();
        later.written_at = "2099-01-01T00:00:00Z".into();
        assert!(later.matches(&a_stamp()));
    }

    #[test]
    fn a_missing_interpreter_forces_a_redo_even_with_a_perfect_stamp() {
        // This is the interrupted-install case the atomic write is there for:
        // the stamp is fine and the runtime is not.
        let dir = tempfile::tempdir().unwrap();
        let layout = crate::layout::Layout::new(dir.path());
        write_atomic(&layout.stamp_path, &a_stamp()).unwrap();
        assert!(needs_bootstrap(&layout, &a_stamp()));
    }

    #[test]
    fn nothing_installed_at_all_forces_a_bootstrap() {
        let dir = tempfile::tempdir().unwrap();
        let layout = crate::layout::Layout::new(dir.path());
        assert!(needs_bootstrap(&layout, &a_stamp()));
    }
}
