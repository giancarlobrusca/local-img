//! Copies the Python side of the app into `src-tauri/python/` so the Tauri
//! bundler can ship it.
//!
//! The alternative — pointing `bundle.resources` at `../../app.py` — asks the
//! bundler to reach outside its own project directory, which it handles
//! inconsistently across the three target platforms. Copying first makes the
//! resource paths boring and identical everywhere.

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
    tauri_build::build();
}
