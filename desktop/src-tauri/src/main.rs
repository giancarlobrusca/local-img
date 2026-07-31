// A GUI app must not also open a console window on Windows.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod bootstrap;
mod cli;
mod commands;
mod events;
mod layout;
mod plan;
mod proc;
mod progress;
mod resources;
mod server;
mod stamp;
mod uninstall;
mod update;

use layout::Layout;
use serde::Serialize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager, State, WebviewWindow};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};
use tauri_plugin_opener::OpenerExt;

/// Everything the shell needs to know, resolved once at startup.
struct Shell {
    layout: Layout,
    outputs: PathBuf,
    resources: PathBuf,
    plan: Result<plan::RuntimePlan, String>,
    server: Mutex<Option<server::Server>>,
    /// Set while a shutdown the user already approved is in progress, so the
    /// close handler does not ask a second time.
    closing: Mutex<bool>,
    /// Set the moment an uninstall starts, so a window that navigates back to
    /// the shell's own page mid-deletion shows the removal screen rather than
    /// offering to set the app up again.
    uninstalling: Mutex<bool>,
    /// The result, once there is one.
    removed: Mutex<Option<events::Removed>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct InitialState {
    needs_setup: bool,
    /// A human-readable size for the welcome copy, or an empty string when
    /// this machine is not supported.
    engine_size: String,
    /// Set when the platform itself is the problem; the welcome screen turns
    /// into the failure screen and the button never appears.
    unsupported: Option<String>,
    /// True from the moment finish_uninstall is called until it is done. The
    /// page reads this because it may well load while the deletion is running.
    uninstalling: bool,
    /// Present once the uninstall has finished. Takes precedence over every
    /// other field: there is nothing left to set up.
    removed: Option<events::Removed>,
}

#[tauri::command]
fn initial_state(shell: State<'_, Shell>) -> InitialState {
    let uninstalling = *shell.uninstalling.lock().unwrap_or_else(|e| e.into_inner());
    let removed = shell.removed.lock().unwrap_or_else(|e| e.into_inner()).clone();
    match &shell.plan {
        Err(reason) => InitialState {
            needs_setup: false,
            engine_size: String::new(),
            unsupported: Some(reason.clone()),
            uninstalling,
            removed,
        },
        Ok(plan) => {
            let requirements = shell.resources.join("requirements.txt");
            let bytes = std::fs::read(&requirements).unwrap_or_default();
            let expected = stamp::Stamp::expected(plan, &bytes);
            InitialState {
                needs_setup: stamp::needs_bootstrap(&shell.layout, &expected),
                engine_size: progress::human_gb(plan.total_download_bytes()),
                unsupported: None,
                uninstalling,
                removed,
            }
        }
    }
}

/// Bootstrap if needed, start the server, hand the window over to it.
///
/// Spawned on a plain thread: everything underneath is blocking, and the
/// webview must stay responsive enough to render the bar.
#[tauri::command]
fn start_setup(app: AppHandle, window: WebviewWindow) {
    std::thread::spawn(move || {
        if let Err(failure) = setup_and_hand_off(&app, &window) {
            let _ = app.emit(events::FAILED, failure);
        }
    });
}

fn setup_and_hand_off(app: &AppHandle, window: &WebviewWindow) -> Result<(), events::Failure> {
    let shell = app.state::<Shell>();

    // An uninstall in progress is deleting data_dir on another thread. The
    // page's own guard against re-entering setup is a convention that lasts
    // exactly until someone edits the HTML; this check is what actually
    // stops `bootstrap::run` from recreating the directory `finish_uninstall`
    // is mid-walk through — which would both resurrect files it just deleted
    // and make `before.saturating_sub(after)` collapse to zero, underreporting
    // what the uninstall freed.
    if *shell.uninstalling.lock().unwrap_or_else(|e| e.into_inner()) {
        return Ok(());
    }

    // Any earlier server is finished with — a retry after a failed handoff,
    // or a reinstall. Shut it down before starting another, or two Python
    // children each holding a multi-GB pipeline overlap until the slot is
    // overwritten.
    if let Ok(mut slot) = shell.server.lock() {
        if let Some(previous) = slot.take() {
            previous.shutdown();
        }
    }

    let plan = shell.plan.as_ref().map_err(|reason| events::Failure {
        title: "This machine is not supported".into(),
        message: reason.clone(),
        diagnostics: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
        retryable: false,
    })?;

    // Checked before anything is downloaded. A bundle that lost a file would
    // otherwise fail as an ImportError inside a child whose stderr nobody sees,
    // after twenty minutes of pip.
    resources::verify(&shell.resources).map_err(|message| events::Failure {
        title: "This installation is incomplete".into(),
        message,
        diagnostics: format!("resources: {}", shell.resources.display()),
        retryable: false,
    })?;

    let requirements = shell.resources.join("requirements.txt");
    let expected = stamp::Stamp::expected(plan, &std::fs::read(&requirements).unwrap_or_default());
    let first_run = stamp::needs_bootstrap(&shell.layout, &expected);

    if first_run {
        let reporter: bootstrap::OnProgress = {
            let app = app.clone();
            Arc::new(move |phase, pct, detail| {
                let _ = app.emit(events::PROGRESS, events::Progress { phase, pct, detail });
            })
        };
        bootstrap::run(plan, &shell.layout, &requirements, &reporter).map_err(|e| {
            events::Failure {
                title: e.title,
                message: e.message,
                diagnostics: e.diagnostics,
                retryable: e.retryable,
            }
        })?;
    }

    let _ = app.emit(
        events::PROGRESS,
        events::Progress {
            phase: "server",
            pct: 0,
            detail: "starting the generation server".into(),
        },
    );

    let started = server::start(&shell.layout, &shell.resources, &shell.outputs)
        .map_err(|e| events::Failure {
            title: "The generation server did not start".into(),
            message: e.message,
            diagnostics: e.diagnostics,
            retryable: true,
        })?;

    // ?firstrun=1 only on the launch that just built the runtime: the wizard
    // has its own reason to appear later (a missing profile), and it should
    // decide that for itself.
    let url = started.url_with_token(first_run);
    // Not `if let Ok(..)`: on a poisoned lock that would skip the store, drop
    // `started`, and Drop would kill the very child we are about to navigate
    // to — silently. A poisoned slot still holds a perfectly good Option.
    *shell
        .server
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = Some(started);
    window
        .navigate(url.parse().expect("a URL we just built"))
        .map_err(|e| events::Failure {
            title: "Could not open the app".into(),
            message: format!("{e}"),
            diagnostics: String::new(),
            retryable: true,
        })?;

    announce_update(app, window);
    Ok(())
}

/// Check for a newer release and tell the page about it.
///
/// Injected rather than served, because a Python route would make the app
/// aware of the shell — which is the one dependency direction this design
/// does not allow.
fn announce_update(app: &AppHandle, window: &WebviewWindow) {
    let window = window.clone();
    let version = app.package_info().version.to_string();
    std::thread::spawn(move || {
        let Some(info) = update::check(&version) else {
            return;
        };
        let Ok(json) = serde_json::to_string(&info) else {
            return;
        };
        // The page may still be loading; it also reads the global on boot, so
        // whichever of the two happens second wins.
        let _ = window.eval(&format!(
            "window.LOCAL_IMG_UPDATE={json};\
             window.dispatchEvent(new Event('local-img-update'));"
        ));
    });
}

#[tauri::command]
fn copy_diagnostics(shell: State<'_, Shell>) -> String {
    let mut out = vec![
        format!("local-img {}", env!("CARGO_PKG_VERSION")),
        format!("platform: {} {}", std::env::consts::OS, std::env::consts::ARCH),
    ];
    match &shell.plan {
        Ok(plan) => {
            out.push(format!("python:  {}", plan.python_url));
            out.push(format!("index:   {}", plan.index_label()));
        }
        Err(reason) => out.push(format!("unsupported: {reason}")),
    }
    out.push(format!("data:    {}", shell.layout.data_dir.display()));
    out.push(format!(
        "stamp:   {}",
        std::fs::read_to_string(&shell.layout.stamp_path)
            .unwrap_or_else(|_| "(none)".into())
            .replace('\n', " ")
    ));
    if let Ok(slot) = shell.server.lock() {
        if let Some(server) = slot.as_ref() {
            out.push(format!("port:    {}", server.port));
            out.push("--- last server output ---".into());
            out.push(server.tail());
        }
    }
    out.join("\n")
}

#[tauri::command]
fn open_log_folder(app: AppHandle, shell: State<'_, Shell>) {
    let _ = app
        .opener()
        .open_path(shell.layout.logs_dir.to_string_lossy().to_string(), None::<&str>);
}

/// Throw the installed runtime away and build it again.
///
/// The stamp already catches an install interrupted partway. This is for the
/// case it cannot catch: an install that completed and is nonetheless broken —
/// a wheel that unpacked badly, a file an antivirus quarantined afterwards.
#[tauri::command]
fn reinstall_engine(app: AppHandle, window: WebviewWindow) {
    {
        let shell = app.state::<Shell>();
        if let Ok(mut slot) = shell.server.lock() {
            if let Some(server) = slot.take() {
                server.shutdown();
            }
        }
        bootstrap::discard(&shell.layout);
    }
    start_setup(app, window);
}

/// The app's own page, which is `tauri://localhost` on macOS and Linux and
/// `http://tauri.localhost` on Windows — the same split `on_navigation` below
/// already has to know about. `useHttpsScheme` is not set in tauri.conf.json,
/// so the Windows scheme is http.
fn local_index_url() -> &'static str {
    #[cfg(windows)]
    {
        "http://tauri.localhost/index.html"
    }
    #[cfg(not(windows))]
    {
        "tauri://localhost/index.html"
    }
}

/// What is left to do once the data directory is gone, per platform.
///
/// The app bundle itself is out of scope: self-deletion needs a helper process
/// that outlives the app on macOS and Linux, for the last 10 MB of 25 GB.
fn last_step() -> (String, String) {
    #[cfg(target_os = "macos")]
    {
        (
            "All that's left is to drag local-img from Applications to the Trash.".into(),
            "Open the Applications folder".into(),
        )
    }
    #[cfg(windows)]
    {
        (
            "All that's left is to remove local-img in Apps & features.".into(),
            "Open Apps & features".into(),
        )
    }
    #[cfg(all(not(target_os = "macos"), not(windows)))]
    {
        (
            "All that's left is to delete the AppImage — or, if you installed the \
             .deb, to run: sudo apt remove local-img"
                .into(),
            "Open the folder it is in".into(),
        )
    }
}

/// The shell half of an uninstall the user has already confirmed.
///
/// Python has deleted the weights, the chunk cache, and — if the box was ticked
/// — the renders, and passes what it freed. It cannot take the last step: it is
/// running from inside `data_dir/runtime`, and Windows does not let a process
/// unlink a file it holds open.
///
/// Every intermediate state here is a valid state, which is why none of this
/// needs a transaction. Interrupted after Python's half: weights gone, runtime
/// present, and the app starts and offers to download a model. Interrupted with
/// the data directory half-deleted: the stamp is gone, so the next launch
/// rebuilds from scratch — precisely what `reinstall_engine` already does.
#[tauri::command]
fn finish_uninstall(app: AppHandle, window: WebviewWindow, freed: u64) {
    {
        let shell = app.state::<Shell>();
        // `closing` stops the close handler asking about a render on a server
        // that is about to stop existing. `uninstalling` is what
        // `setup_and_hand_off` checks before it touches `shell.server`, so a
        // page that re-enters setup while this runs is refused rather than
        // recreating `data_dir` out from under the deletion below.
        *shell.closing.lock().unwrap_or_else(|e| e.into_inner()) = true;
        *shell.uninstalling.lock().unwrap_or_else(|e| e.into_inner()) = true;
    }

    // Back to the shell's own page before anything is deleted. The page Python
    // served is about to lose its server, and removing a 7 GB runtime takes
    // long enough that doing it first would leave a dead window on screen.
    let _ = window.navigate(
        local_index_url()
            .parse()
            .expect("a URL compiled into the binary"),
    );

    // On a thread for the same reason `start_setup` is: everything below
    // blocks, and the webview has a screen to draw.
    std::thread::spawn(move || {
        let shell = app.state::<Shell>();
        // Not `if let Ok(..)`: on a poisoned lock that would silently skip
        // shutting Python down, and `remove_data_dir` would then delete the
        // runtime out from under a still-live process — on Windows every open
        // file lands in `resisted`, and the removed screen lists hundreds of
        // paths instead of the handful a real failure would leave.
        {
            let mut slot = shell.server.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(server) = slot.take() {
                server.shutdown();
            }
        }

        // Measured either side rather than trusted: the total on screen has to
        // be what came off the disk, including when part of it would not go.
        let before = progress::dir_size(&shell.layout.data_dir);
        let resisted = uninstall::remove_data_dir(&shell.layout.data_dir);
        let after = progress::dir_size(&shell.layout.data_dir);

        let (last_step, open_label) = last_step();
        let removed = events::Removed {
            freed: progress::human_gb(freed.saturating_add(before.saturating_sub(after))),
            resisted,
            data_dir: shell.layout.data_dir.display().to_string(),
            last_step,
            open_label,
        };

        *shell.removed.lock().unwrap_or_else(|e| e.into_inner()) = Some(removed.clone());
        *shell.uninstalling.lock().unwrap_or_else(|e| e.into_inner()) = false;
        // The page may have finished loading before or after this point; it
        // reads `initial_state` on boot as well, so whichever happens second wins.
        let _ = app.emit(events::REMOVED, removed);
    });
}

/// Open the place the app itself has to be removed from.
#[tauri::command]
fn open_app_location(app: AppHandle) {
    #[cfg(windows)]
    {
        // An .msi install is removed in Apps & features, not by dragging a
        // folder, so this opens the settings page rather than a directory.
        let _ = app.opener().open_url("ms-settings:appsfeatures", None::<&str>);
    }
    #[cfg(not(windows))]
    {
        let _ = app.opener().open_path(
            uninstall::app_folder().to_string_lossy().to_string(),
            None::<&str>,
        );
    }
}

/// Open the data directory, for the case where part of it would not delete.
///
/// Beside `open_log_folder`, which points one level deeper. The removed card
/// shows this button only when something resisted — it exists so the user can
/// go and look at the exact files named on that screen.
#[tauri::command]
fn open_data_dir(app: AppHandle, shell: State<'_, Shell>) {
    let _ = app.opener().open_path(
        shell.layout.data_dir.to_string_lossy().to_string(),
        None::<&str>,
    );
}

#[tauri::command]
fn quit_app(app: AppHandle) {
    app.exit(0);
}

fn main() {
    // Handled before Tauri exists, and this is not a convenience: a Linux CI
    // runner has no display, so constructing the webview would fail long
    // before the bootstrap — the thing under test — got to run.
    if std::env::args().any(|a| a == "--bootstrap-only") {
        std::process::exit(cli::bootstrap_only());
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        // Two instances would load two 7 GB pipelines and take the machine
        // down with them. A second launch focuses the first one's window.
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.unminimize();
                let _ = window.set_focus();
            }
        }))
        .invoke_handler(tauri::generate_handler![
            initial_state,
            start_setup,
            reinstall_engine,
            copy_diagnostics,
            open_log_folder,
            finish_uninstall,
            open_app_location,
            open_data_dir,
            quit_app
        ])
        .setup(|app| {
            // local_data_dir is exactly the spec's three paths, without the
            // bundle identifier Tauri's app_* helpers append:
            //   ~/Library/Application Support | %LOCALAPPDATA% | ~/.local/share
            let data_dir = app.path().local_data_dir()?.join("local-img");
            // Renders go to Pictures because "your images are in Pictures" is
            // the only location that needs no explanation.
            let outputs = app.path().picture_dir()?.join("local-img");
            let resources = app.path().resource_dir()?.join("python");
            std::fs::create_dir_all(&data_dir).ok();
            std::fs::create_dir_all(&outputs).ok();

            app.manage(Shell {
                layout: Layout::new(&data_dir),
                outputs,
                resources,
                plan: plan::current(),
                server: Mutex::new(None),
                closing: Mutex::new(false),
                uninstalling: Mutex::new(false),
                removed: Mutex::new(None),
            });

            let handle = app.handle().clone();
            tauri::WebviewWindowBuilder::new(
                app,
                "main",
                tauri::WebviewUrl::App("index.html".into()),
            )
            .title("local-img")
            .inner_size(1180.0, 800.0)
            .min_inner_size(900.0, 600.0)
            // The update banner is a plain link. Rather than navigating the
            // app's own window to GitHub, hand anything that is not our own
            // server to the system browser.
            .on_navigation(move |url| {
                // The app's own page is `tauri://localhost` on macOS and Linux
                // but `http(s)://tauri.localhost` on Windows and Android, so
                // the host has to be checked as well as the scheme — matching
                // on the scheme alone would hand Windows' very first
                // navigation, index.html itself, to the system browser.
                let local_host = matches!(
                    url.host_str(),
                    Some("127.0.0.1") | Some("tauri.localhost") | Some("ipc.localhost")
                );
                if matches!(url.scheme(), "tauri" | "asset" | "ipc" | "http+tauri")
                    || local_host
                {
                    return true;
                }
                let _ = handle.opener().open_url(url.as_str(), None::<&str>);
                false
            })
            .build()?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if !matches!(event, tauri::WindowEvent::CloseRequested { .. }) {
                return;
            }
            let app = window.app_handle();
            let shell = app.state::<Shell>();

            // A poisoned bool is still a perfectly good bool, and panicking
            // here would take the event loop down with it.
            if *shell.closing.lock().unwrap_or_else(|e| e.into_inner()) {
                return;
            }

            // The image is written only when the render finishes, so closing
            // four minutes into a flux job loses it entirely.
            //
            // Copy what the probe needs, then release the lock — holding it
            // across an HTTP call would block copy_diagnostics for as long as
            // the timeout.
            let probe = shell
                .server
                .lock()
                .ok()
                .and_then(|slot| slot.as_ref().map(|s| (s.port, s.token.clone())));
            let generating = probe
                .map(|(port, token)| server::is_generating_at(port, &token))
                .unwrap_or(false);

            if generating {
                // True for the FIRST button, which is "Close anyway" — so this
                // is the user electing to lose the render, not to keep it.
                let close_anyway = app
                    .dialog()
                    .message(
                        "An image is still being generated. It is only saved when it \
                         finishes, so closing now loses it.",
                    )
                    .title("Still generating")
                    .kind(MessageDialogKind::Warning)
                    .buttons(MessageDialogButtons::OkCancelCustom(
                        "Close anyway".into(),
                        "Keep generating".into(),
                    ))
                    // blocking_show on the event-loop thread is against the
                    // plugin's own guidance, and is safe here only because rfd
                    // presents a PARENTLESS message dialog off the app's loop on
                    // all three targets (macOS CFUserNotificationDisplayAlert on
                    // its own thread, Windows ThreadFuture, Linux GtkGlobalThread).
                    // That is an implementation detail, not a contract — if this
                    // ever deadlocks after a tauri-plugin-dialog or rfd upgrade,
                    // this is the line, and the fix is to move the prompt off the
                    // event-loop thread rather than to fight the dialog.
                    .blocking_show();
                if !close_anyway {
                    if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                        api.prevent_close();
                    }
                    return;
                }
            }

            *shell.closing.lock().unwrap_or_else(|e| e.into_inner()) = true;
            // The semicolon matters: as a tail expression the guard's temporary
            // would outlive `shell` and the borrow checker rejects it.
            if let Ok(mut slot) = shell.server.lock() {
                if let Some(server) = slot.take() {
                    server.shutdown();
                }
            };
        })
        .run(tauri::generate_context!())
        .expect("error while running local-img");
}
