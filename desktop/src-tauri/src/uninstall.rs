//! Deleting the data directory, once nothing is running from inside it.
//!
//! This is the step Python cannot take for itself. `app.py` runs out of
//! `data_dir/runtime`, and Windows does not let a process unlink a file it
//! holds open — so the shell shuts the server down and removes the directory
//! afterwards. On Unix it could have gone the other way, and having one code
//! path rather than two is worth more than the round trip costs.

//! Measuring is not here: `progress::dir_size` already walks a tree the way
//! this needs — recursively, ignoring what it cannot read, and measuring a
//! symlink rather than its target. `finish_uninstall` calls that either side of
//! the deletion below.

use std::path::{Path, PathBuf};

/// Delete `data_dir` and everything under it. Returns the paths that resisted.
///
/// Not `std::fs::remove_dir_all`, which stops at the first failure and cannot
/// say which file it was. What this needs instead is for one locked file to
/// cost exactly that file, and for the summary afterwards to name it — nothing
/// here pretends to a success it did not have.
// Dead from this crate's point of view until Task 5 wires it into a Tauri
// command; the tests below are its only caller for now.
#[allow(dead_code)]
pub fn remove_data_dir(data_dir: &Path) -> Vec<String> {
    let mut resisted = Vec::new();
    if std::fs::symlink_metadata(data_dir).is_err() {
        return resisted;        // never installed, or already gone
    }
    remove_tree(data_dir, &mut resisted);
    resisted
}

fn remove_tree(path: &Path, resisted: &mut Vec<String>) {
    // symlink_metadata, not metadata: a symlink is unlinked, never descended
    // into. Following one would delete whatever it points at, which is the one
    // way a bounded deletion escapes its bound.
    let Ok(meta) = std::fs::symlink_metadata(path) else {
        resisted.push(path.display().to_string());
        return;
    };

    if meta.is_dir() {
        match std::fs::read_dir(path) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    remove_tree(&entry.path(), resisted);
                }
            }
            Err(_) => {
                resisted.push(path.display().to_string());
                return;
            }
        }
        if std::fs::remove_dir(path).is_err() {
            resisted.push(path.display().to_string());
        }
    } else if std::fs::remove_file(path).is_err() {
        resisted.push(path.display().to_string());
    }
}

/// The folder the user has to remove the app itself from.
// Dead from this crate's point of view until Task 5 wires it into a Tauri
// command; no test here calls the impure half either, since current_exe()
// only reflects the test binary. app_folder_from below carries the coverage.
#[allow(dead_code)]
pub fn app_folder() -> PathBuf {
    app_folder_from(&std::env::current_exe().unwrap_or_default())
}

/// The pure half of `app_folder`, so the macOS case is testable off macOS.
///
/// Inside a bundle `current_exe` is `local-img.app/Contents/MacOS/local-img`,
/// so the folder that holds the *bundle* is four levels up — opening the folder
/// that holds the binary would open the inside of the bundle, where there is
/// nothing to drag anywhere. Everywhere else the binary sits in the folder
/// directly, and no ancestor ends in `.app`, so this reduces to the parent.
pub fn app_folder_from(exe: &Path) -> PathBuf {
    let bundle = exe
        .ancestors()
        .find(|p| p.extension().is_some_and(|ext| ext == "app"));
    let folder = match bundle {
        Some(bundle) => bundle.parent(),
        None => exe.parent(),
    };
    folder.unwrap_or(Path::new("")).to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn a_data_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("local-img");
        std::fs::create_dir_all(data.join("runtime").join("python").join("bin")).unwrap();
        std::fs::create_dir_all(data.join("logs")).unwrap();
        std::fs::write(data.join("runtime/python/bin/python3"), vec![b'x'; 500]).unwrap();
        std::fs::write(data.join("logs/server.log"), b"hello").unwrap();
        std::fs::write(data.join("runtime.json"), b"{}").unwrap();
        dir
    }

    #[test]
    fn the_whole_data_directory_goes() {
        let dir = a_data_dir();
        let data = dir.path().join("local-img");
        assert!(remove_data_dir(&data).is_empty(), "nothing should resist");
        assert!(!data.exists(), "the data directory itself is removed, not emptied");
    }

    #[test]
    fn nothing_outside_the_data_directory_is_touched() {
        // layout.rs already asserts every path it derives sits under data_dir.
        // This is the other half of that claim: the deletion respects it.
        let dir = a_data_dir();
        let data = dir.path().join("local-img");
        let sibling = dir.path().join("something-else");
        std::fs::create_dir_all(&sibling).unwrap();
        std::fs::write(sibling.join("keep.txt"), b"keep").unwrap();

        remove_data_dir(&data);

        assert!(sibling.join("keep.txt").exists(), "a sibling directory survives");
        assert_eq!(std::fs::read(sibling.join("keep.txt")).unwrap(), b"keep");
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_out_of_the_directory_is_unlinked_not_followed() {
        // Following one would delete whatever it points at, which is the one
        // way a bounded deletion escapes its bound.
        let dir = a_data_dir();
        let data = dir.path().join("local-img");
        let outside = dir.path().join("precious.txt");
        std::fs::write(&outside, b"precious").unwrap();
        std::os::unix::fs::symlink(&outside, data.join("shortcut")).unwrap();

        assert!(remove_data_dir(&data).is_empty());
        assert!(!data.exists());
        assert!(outside.exists(), "the symlink's target survives");
    }

    #[test]
    fn a_directory_that_is_not_there_is_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let absent = dir.path().join("never-installed");
        assert!(remove_data_dir(&absent).is_empty());
        assert!(!absent.exists());
    }

    #[test]
    fn the_measurement_this_pairs_with_already_exists() {
        // finish_uninstall measures either side of the deletion with
        // progress::dir_size, which predates this module and is tested there.
        // This asserts only the pairing — that what remove_data_dir leaves
        // behind is what that function reports.
        let dir = a_data_dir();
        let data = dir.path().join("local-img");
        // 500 + 5 + 2 across the three files planted above.
        assert_eq!(crate::progress::dir_size(&data), 507);
        remove_data_dir(&data);
        assert_eq!(crate::progress::dir_size(&data), 0, "an absent tree measures zero");
    }

    #[cfg(unix)]
    #[test]
    fn what_resists_is_named_and_the_rest_still_goes() {
        // A read-only directory cannot have entries unlinked from it — the
        // portable stand-in for a file an antivirus has locked. The summary has
        // to say what actually happened; a dialog reading 23 GB when it freed
        // 19 is worse than no button at all.
        use std::os::unix::fs::PermissionsExt;
        let dir = a_data_dir();
        let data = dir.path().join("local-img");
        let logs = data.join("logs");
        std::fs::set_permissions(&logs, std::fs::Permissions::from_mode(0o500)).unwrap();

        let resisted = remove_data_dir(&data);

        std::fs::set_permissions(&logs, std::fs::Permissions::from_mode(0o700)).unwrap();
        assert!(!resisted.is_empty(), "the locked file is reported");
        assert!(
            resisted.iter().any(|p| p.contains("server.log")),
            "by name: {resisted:?}"
        );
        assert!(
            !data.join("runtime").exists(),
            "everything that could go, went"
        );
    }

    #[test]
    fn a_mac_bundle_resolves_to_the_folder_that_holds_it() {
        // current_exe inside a .app is four levels deep. Opening the folder
        // that holds the *binary* would open the inside of the bundle, where
        // there is nothing for the user to drag anywhere.
        assert_eq!(
            app_folder_from(Path::new(
                "/Applications/local-img.app/Contents/MacOS/local-img"
            )),
            Path::new("/Applications")
        );
    }

    #[test]
    fn a_plain_binary_resolves_to_its_own_folder() {
        assert_eq!(
            app_folder_from(Path::new("/opt/local-img/local-img")),
            Path::new("/opt/local-img")
        );
        assert_eq!(
            app_folder_from(Path::new("/local-img")),
            Path::new("/")
        );
    }
}
