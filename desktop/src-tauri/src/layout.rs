//! Where each part of the private runtime sits under the data directory.
//!
//! Every path here is derived, never searched for. The two platform branches
//! reflect exactly one fact: python-build-standalone lays Windows out
//! differently from Unix, with the interpreter at the root and site-packages
//! under `Lib` rather than `lib/python3.12`.

use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Layout {
    pub data_dir: PathBuf,
    pub runtime_dir: PathBuf,
    pub interpreter: PathBuf,
    pub site_packages: PathBuf,
    pub logs_dir: PathBuf,
    /// Beside the runtime, not inside it: a half-extracted archive must never
    /// be able to bury a stale stamp and make a broken runtime look current.
    pub stamp_path: PathBuf,
    /// Where the downloaded archive lands before it is unpacked.
    pub archive_path: PathBuf,
}

impl Layout {
    pub fn new(data_dir: &Path) -> Self {
        let runtime_dir = data_dir.join("runtime");
        // Every install_only archive has a single top-level `python/`.
        let python_root = runtime_dir.join("python");

        #[cfg(windows)]
        let (interpreter, site_packages) = (
            python_root.join("python.exe"),
            python_root.join("Lib").join("site-packages"),
        );
        #[cfg(not(windows))]
        let (interpreter, site_packages) = (
            python_root.join("bin").join("python3"),
            python_root
                .join("lib")
                .join(format!("python{}", crate::plan::PYTHON_MINOR))
                .join("site-packages"),
        );

        Self {
            data_dir: data_dir.to_path_buf(),
            runtime_dir,
            interpreter,
            site_packages,
            logs_dir: data_dir.join("logs"),
            stamp_path: data_dir.join("runtime.json"),
            archive_path: data_dir.join("python-runtime.tar.gz"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn everything_sits_under_the_data_directory() {
        let l = Layout::new(Path::new("/data"));
        for path in [&l.runtime_dir, &l.interpreter,
                     &l.site_packages, &l.logs_dir, &l.stamp_path, &l.archive_path] {
            assert!(path.starts_with("/data"), "escaped: {}", path.display());
        }
    }

    #[test]
    fn the_stamp_sits_beside_the_runtime_not_inside_it() {
        // Inside, a half-extracted archive could bury a stale stamp and make a
        // broken runtime look current.
        let l = Layout::new(Path::new("/data"));
        assert_eq!(l.stamp_path, Path::new("/data/runtime.json"));
        assert!(!l.stamp_path.starts_with("/data/runtime/"));
    }

    #[test]
    fn the_archive_unpacks_into_a_python_directory() {
        // python-build-standalone's install_only archives all have a single
        // top-level `python/` entry.
        let l = Layout::new(Path::new("/data"));
        let python_root = l.runtime_dir.join("python");
        assert!(l.interpreter.starts_with(&python_root));
        assert!(l.site_packages.starts_with(&python_root));
    }

    #[cfg(unix)]
    #[test]
    fn unix_layout_matches_the_archive() {
        let l = Layout::new(Path::new("/data"));
        assert_eq!(l.interpreter, Path::new("/data/runtime/python/bin/python3"));
        assert_eq!(
            l.site_packages,
            Path::new("/data/runtime/python/lib/python3.12/site-packages")
        );
    }

    #[cfg(windows)]
    #[test]
    fn windows_layout_matches_the_archive() {
        let l = Layout::new(Path::new("C:\\data"));
        assert!(l.interpreter.ends_with("runtime\\python\\python.exe"));
        assert!(l.site_packages.ends_with("runtime\\python\\Lib\\site-packages"));
    }
}
