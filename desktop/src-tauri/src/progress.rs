//! Turning bytes on disk into a bar, and answering whether there is room.

use crate::plan::RuntimePlan;
use std::path::Path;

/// A sampled percentage, capped at 99.
///
/// The cap is the point: `expected` is an estimate, so reaching it proves
/// nothing. 100 is emitted by the caller when the step actually finishes, and
/// a bar that sits at 99 for a moment reads far better than one that sits at
/// 100 for a minute.
pub fn pct(done: u64, expected: u64) -> u8 {
    if expected == 0 {
        return 0;
    }
    let raw = (done as f64 / expected as f64) * 100.0;
    raw.clamp(0.0, 99.0) as u8
}

/// Total size of everything under `path`, ignoring anything unreadable.
///
/// Sampled while pip is writing into the directory, so it is a moving target
/// by construction — an I/O error on one entry must lower the number, never
/// abort the walk.
pub fn dir_size(path: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    let mut total = 0;
    for entry in entries.flatten() {
        match entry.file_type() {
            Ok(t) if t.is_dir() => total += dir_size(&entry.path()),
            // Metadata on a symlink reports the link, not its target, which is
            // what we want: counting targets would double-count.
            Ok(_) => total += entry.metadata().map(|m| m.len()).unwrap_or(0),
            Err(_) => {}
        }
    }
    total
}

/// How much free space the bootstrap needs before it starts.
///
/// The archive and the tree it unpacks to coexist on disk for a moment, and
/// pip keeps a wheel cache alongside what it installs. The 15% margin is the
/// same headroom app.py already applies to model downloads.
pub fn required_disk_bytes(plan: &RuntimePlan) -> u64 {
    let raw = plan.python_bytes * 4 + plan.expected_site_packages;
    (raw as f64 * 1.15) as u64
}

pub fn free_disk_bytes(path: &Path) -> Option<u64> {
    fs4::available_space(path).ok()
}

pub fn human_gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1e9)
}

/// A size in whichever unit suits it.
///
/// The Python archive is 25–111 MB; the dependency install is 1–7 GB. Both are
/// reported through the same progress line, so fixing the unit at GB makes the
/// entire download phase read "0.0 GB of 0.0 GB" — which is the first text a
/// new user ever sees from this app. Use `human_gb` where the quantity is
/// always large (disk requirements); use this where it varies.
pub fn human_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1e9)
    } else {
        format!("{:.0} MB", bytes as f64 / 1e6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pct_covers_its_edges() {
        assert_eq!(pct(0, 1_000), 0);
        assert_eq!(pct(500, 1_000), 50);
        assert_eq!(pct(990, 1_000), 99);
        assert_eq!(pct(1_000, 1_000), 99, "a sampled bar never claims completion");
        assert_eq!(pct(9_000, 1_000), 99, "an underestimate must not exceed 99");
        assert_eq!(pct(100, 0), 0, "a zero expectation is unknowable, not infinite");
        assert_eq!(pct(0, 0), 0);
    }

    #[test]
    fn pct_rises_monotonically() {
        let mut last = 0;
        for done in (0..=1_000u64).step_by(37) {
            let now = pct(done, 1_000);
            assert!(now >= last, "went backwards at {done}");
            last = now;
        }
    }

    #[test]
    fn dir_size_sums_recursively() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a"), vec![0u8; 100]).unwrap();
        std::fs::create_dir(dir.path().join("nested")).unwrap();
        std::fs::write(dir.path().join("nested/b"), vec![0u8; 250]).unwrap();
        assert_eq!(dir_size(dir.path()), 350);
    }

    #[test]
    fn dir_size_of_an_absent_directory_is_zero() {
        // Sampled while pip is still creating it, this must be 0, not a panic.
        assert_eq!(dir_size(std::path::Path::new("/zz-no-such-directory")), 0);
    }

    #[test]
    fn the_disk_requirement_covers_the_archive_and_its_unpacking() {
        let plan = crate::plan::plan_for("linux", "x86_64", true).unwrap();
        let need = required_disk_bytes(&plan);
        assert!(need > plan.expected_site_packages + plan.python_bytes,
                "the archive and the tree it unpacks to coexist on disk");
        assert!(need < plan.expected_site_packages * 2);
    }

    #[test]
    fn free_disk_answers_for_a_real_directory() {
        let dir = tempfile::tempdir().unwrap();
        assert!(free_disk_bytes(dir.path()).unwrap() > 0);
    }

    // Unix only, and the asymmetry is the platform's, not a gap in the test.
    // A path that does not exist has no volume to report on here, so the
    // answer is None. On Windows the same string is read relative to the
    // current drive, resolves to that drive's root, and reports its free
    // space — which is the answer the bootstrap actually wants, since it asks
    // about a directory it is *about* to create. Asserting None there would
    // demand behaviour that would be wrong.
    #[cfg(unix)]
    #[test]
    fn free_disk_has_no_answer_for_a_path_with_no_volume() {
        assert!(free_disk_bytes(std::path::Path::new("/zz-nope")).is_none());
    }

    #[test]
    fn sizes_read_as_gigabytes() {
        assert_eq!(human_gb(0), "0.0 GB");
        assert_eq!(human_gb(1_500_000_000), "1.5 GB");
        assert_eq!(human_gb(7_000_000_000), "7.0 GB");
    }

    #[test]
    fn human_size_picks_the_unit_that_suits_the_number() {
        // The download phase is tens of megabytes; rendering it in GB would
        // show "0.0 MB of 0.0 GB" for its whole duration.
        assert_eq!(human_size(25_149_265), "25 MB");
        assert_eq!(human_size(111_358_187), "111 MB");
        assert_eq!(human_size(1_500_000_000), "1.5 GB");
        assert_eq!(human_size(0), "0 MB");
    }
}
