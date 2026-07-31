//! Copies the Python side of the app into `src-tauri/python/` so the Tauri
//! bundler can ship it.
//!
//! The alternative — pointing `bundle.resources` at `../../app.py` — asks the
//! bundler to reach outside its own project directory, which it handles
//! inconsistently across the three target platforms. Copying first makes the
//! resource paths boring and identical everywhere.
//!
//! It also declares the app's ACL manifest, which is what lets a capability
//! grant one of this crate's own commands to the page the Python server hosts.
//! Note the consequence: once an app manifest exists, every command needs a
//! grant, including from the shell's own window. `src/commands.rs` holds the
//! list and its tests hold the two capability files to it.

include!("src/commands.rs");
include!("src/resources.rs");

fn main() {
    let dest = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("python");
    for name in PYTHON_FILES {
        let from = repo_root().join(name);
        let to = dest.join(name);
        std::fs::create_dir_all(to.parent().unwrap()).expect("create resource directory");
        std::fs::copy(&from, &to)
            .unwrap_or_else(|e| panic!("copy {} -> {}: {e}", from.display(), to.display()));
        println!("cargo:rerun-if-changed={}", from.display());
    }
    println!("cargo:rerun-if-changed=src/commands.rs");

    tauri_build::try_build(
        tauri_build::Attributes::new()
            .app_manifest(tauri_build::AppManifest::new().commands(COMMANDS)),
    )
    .expect("failed to run tauri-build");
}
