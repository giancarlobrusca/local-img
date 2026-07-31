// A GUI app must not also open a console window on Windows.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod bootstrap;
mod cli;
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
}

#[tauri::command]
fn initial_state(shell: State<'_, Shell>) -> InitialState {
    match &shell.plan {
        Err(reason) => InitialState {
            needs_setup: false,
            engine_size: String::new(),
            unsupported: Some(reason.clone()),
        },
        Ok(plan) => {
            let requirements = shell.resources.join("requirements.txt");
            let bytes = std::fs::read(&requirements).unwrap_or_default();
            let expected = stamp::Stamp::expected(plan, &bytes);
            InitialState {
                needs_setup: stamp::needs_bootstrap(&shell.layout, &expected),
                engine_size: progress::human_gb(plan.total_download_bytes()),
                unsupported: None,
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
            open_log_folder
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
