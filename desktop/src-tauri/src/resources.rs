// The Python side of the app, as a list.
//
// `build.rs` copies these into `src-tauri/python/` so the bundler can pick
// them up, and the runtime reads them back out of the resource directory. A
// bundle that silently ships without `models.py` fails on a user's machine
// and nowhere else, so the list is a tested constant rather than a glob.
//
// Paths are relative to the repo root and use forward slashes on every
// platform — they are joined, never parsed.
//
// NOTE: this is a `//` comment, not a `//!` module doc, because build.rs
// pulls this file in with `include!`, which splices it in after build.rs's
// own doc comment — not at the true start of a file — and rustc rejects an
// inner doc comment (E0753) anywhere but the very top of a crate or module.

pub const PYTHON_FILES: &[&str] = &[
    "app.py",
    "paths.py",
    "hardware.py",
    "models.py",
    "download.py",
    "requirements.txt",
    "web/index.html",
];

/// The repo root, two levels above `src-tauri/`.
///
/// Resolved from `CARGO_MANIFEST_DIR` so it is correct in `build.rs`, in
/// `cargo test`, and in `cargo run` alike. It is deliberately *not* how the
/// installed app finds its files — that goes through the resource directory.
// Dead from the binary's point of view and deliberately kept: build.rs pulls
// this file in with `include!` and calls it to copy the resources, and the
// tests below use it. Deleting it would break the build script.
#[allow(dead_code)]
pub fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

/// Whether a resource directory actually holds everything, and what is absent
/// if not.
///
/// Run at startup rather than trusted. A bundle that lost a file fails
/// somewhere unhelpful otherwise — an ImportError inside a child process whose
/// stderr the user never sees.
pub fn verify(dir: &std::path::Path) -> Result<(), String> {
    let missing: Vec<&str> = PYTHON_FILES
        .iter()
        .copied()
        .filter(|name| !dir.join(name).is_file())
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "this installation is incomplete — it is missing {}. Reinstalling \
             the app is the fix.",
            missing.join(", ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_listed_file_exists_in_the_repo() {
        for name in PYTHON_FILES {
            let path = repo_root().join(name);
            assert!(path.is_file(), "missing bundled resource: {}", path.display());
        }
    }

    #[test]
    fn the_entry_point_and_its_seam_are_both_listed() {
        // app.py imports paths.py at module scope; shipping one without the
        // other is an ImportError on the user's first launch.
        assert!(PYTHON_FILES.contains(&"app.py"));
        assert!(PYTHON_FILES.contains(&"paths.py"));
    }

    #[test]
    fn every_module_app_imports_is_listed() {
        // Parsed rather than asserted by hand: a future `import foo` in app.py
        // that nobody adds here would otherwise ship broken.
        let source = std::fs::read_to_string(repo_root().join("app.py")).unwrap();
        for line in source.lines() {
            let line = line.trim();
            let module = line
                .strip_prefix("import ")
                .or_else(|| line.strip_prefix("from ").and_then(|r| r.split(' ').next()))
                .map(str::trim);
            let Some(module) = module else { continue };
            let candidate = format!("{module}.py");
            // Only local modules matter: torch and fastapi come from pip.
            if repo_root().join(&candidate).is_file() {
                assert!(
                    PYTHON_FILES.contains(&candidate.as_str()),
                    "app.py imports {module} but it is not bundled"
                );
            }
        }
    }

    #[test]
    fn no_path_escapes_the_repo() {
        for name in PYTHON_FILES {
            assert!(!name.starts_with('/'), "{name} is absolute");
            assert!(!name.contains(".."), "{name} escapes the repo root");
        }
    }

    #[test]
    fn verify_names_what_is_missing() {
        assert!(verify(&repo_root()).is_ok());
        let complaint = verify(std::path::Path::new("/zz-not-a-bundle")).unwrap_err();
        assert!(complaint.contains("app.py"), "got: {complaint}");
    }
}
