// Every `#[tauri::command]` in this crate, as a list.
//
// `build.rs` includes this file and hands the list to tauri-build, which
// autogenerates one ACL permission per command. That has a consequence worth
// stating plainly: before an app manifest exists, an unregistered command
// simply works; after one exists, every app command is ACL-checked, and one
// missing grant is a runtime rejection with no compile error anywhere. Hence
// the tests below, which compare this list against `main.rs` and against both
// capability files.
//
// NOTE: this is a `//` comment, not a `//!` module doc, for the same reason as
// `resources.rs` — build.rs splices this file in with `include!`, and rustc
// rejects an inner doc comment anywhere but the very top of a module.

// Dead from the binary's point of view: build.rs pulls this file in with
// `include!` and hands the list to tauri-build in a wholly separate
// compilation, and the tests below are its other reader. Nothing in the
// binary itself calls it directly, which is what `dead_code` would otherwise
// complain about — the same shape as `repo_root` in `resources.rs`.
#[allow(dead_code)]
pub const COMMANDS: &[&str] = &[
    "initial_state",
    "start_setup",
    "reinstall_engine",
    "copy_diagnostics",
    "open_log_folder",
    "finish_uninstall",
    "open_app_location",
    "open_data_dir",
    "quit_app",
];

/// The commands the page Python serves may call.
///
/// One, and it is the second half of an uninstall the user has already
/// confirmed. The served page gaining IPC at all is the concession this design
/// makes; it does not also get the core API, the dialog plugin, or the opener.
// Read by the tests below and by nothing in the binary itself, which is what
// `dead_code` would otherwise complain about — including inside build.rs,
// which pulls this file in with `include!`.
#[allow(dead_code)]
pub const LOCAL_PAGE_COMMANDS: &[&str] = &["finish_uninstall"];

#[cfg(test)]
mod tests {
    use super::*;

    fn crate_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    fn capability(name: &str) -> String {
        std::fs::read_to_string(crate_dir().join("capabilities").join(name))
            .unwrap_or_else(|e| panic!("read capabilities/{name}: {e}"))
    }

    fn permission(command: &str) -> String {
        format!("\"allow-{}\"", command.replace('_', "-"))
    }

    #[test]
    fn the_list_matches_what_main_registers() {
        // Parsed rather than asserted by hand: with an app manifest declared, a
        // command registered in one place and not the other fails on a user's
        // machine and nowhere else.
        let main = std::fs::read_to_string(crate_dir().join("src").join("main.rs")).unwrap();
        let block = main
            .split("tauri::generate_handler![")
            .nth(1)
            .expect("main.rs registers a handler")
            .split(']')
            .next()
            .unwrap();
        let registered: Vec<&str> = block
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();

        for command in COMMANDS {
            assert!(
                registered.contains(command),
                "{command} is in COMMANDS but not in generate_handler!"
            );
        }
        assert_eq!(
            registered.len(),
            COMMANDS.len(),
            "generate_handler! registers {registered:?}, COMMANDS lists {COMMANDS:?}"
        );
    }

    #[test]
    fn the_default_capability_grants_every_command() {
        // The shell's own first-run window used every one of these before any
        // ACL existed. Declaring the manifest without this would break setup.
        let json = capability("default.json");
        for command in COMMANDS {
            assert!(
                json.contains(&permission(command)),
                "default.json is missing {}",
                permission(command)
            );
        }
    }

    #[test]
    fn the_served_page_gets_exactly_the_commands_it_is_supposed_to() {
        let json = capability("local-page.json");
        for command in COMMANDS {
            assert_eq!(
                json.contains(&permission(command)),
                LOCAL_PAGE_COMMANDS.contains(command),
                "local-page.json and LOCAL_PAGE_COMMANDS disagree about {command}"
            );
        }
    }

    #[test]
    fn the_served_page_capability_is_scoped_to_the_loopback_origin() {
        let json = capability("local-page.json");
        assert!(
            json.contains("\"http://127.0.0.1:*\""),
            "the origin is the loopback address the server binds, with any port"
        );
        assert!(
            !json.contains("core:default"),
            "the served page gets one command, not the core API"
        );
        assert!(
            !json.contains("opener:default") && !json.contains("dialog:default"),
            "no plugin reaches the served page"
        );
    }

    #[test]
    fn the_capability_pattern_matches_the_url_the_shell_actually_navigates_to() {
        // The property the whole uninstall depends on, and the one that fails
        // silently: if this pattern does not match the page's real URL, the
        // ACL rejects `finish_uninstall` at runtime with no compile error and
        // no failing test anywhere. Until now it was argued from reading
        // urlpattern's source — that a pattern with an empty pathname, search
        // and hash has each replaced with `*`. This executes it instead,
        // against the URL `server::handoff_url` genuinely builds, so a later
        // edit to either side is caught here rather than on a user's machine.
        use tauri::utils::acl::RemoteUrlPattern;

        let json = capability("local-page.json");
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let raw = value["remote"]["urls"][0].as_str().expect("a remote url");
        let pattern: RemoteUrlPattern = raw.parse().expect("the pattern parses");

        // Both shapes the shell hands the window: the first launch carries
        // ?firstrun=1, every later one does not. The token is always present.
        for url in [
            crate::server::handoff_url(51234, "abc123", true),
            crate::server::handoff_url(7788, "deadbeef", false),
        ] {
            let parsed = url.parse().expect("a URL we just built");
            assert!(pattern.test(&parsed), "{raw} must match {url}");
        }

        // And it must not be so loose that any page can reach the command.
        for foreign in [
            "http://evil.example/",
            "https://127.0.0.1:51234/",
            "http://127.0.0.2:51234/",
        ] {
            let parsed = foreign.parse().unwrap();
            assert!(!pattern.test(&parsed), "{raw} must not match {foreign}");
        }
    }

    #[test]
    fn build_rs_declares_the_manifest_from_this_list() {
        let build = std::fs::read_to_string(crate_dir().join("build.rs")).unwrap();
        assert!(
            build.contains("app_manifest") && build.contains("COMMANDS"),
            "build.rs must generate the permissions from COMMANDS, or the \
             capability files reference identifiers that do not exist"
        );
    }
}
