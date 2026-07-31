//! Building the private Python runtime.
//!
//! The order is fixed and the reason is in the spec: hardware cannot be
//! measured before the runtime exists, because `hardware._budget_gib()` asks
//! torch how much memory Metal or CUDA will hand out. So the first launch is
//! necessarily two downloads in sequence, and only the first one's size is
//! known in advance.
//!
//! Nothing here is unit-tested. Downloading, extracting and running pip are
//! covered by the CI smoke job on all three platforms; the pieces that *can*
//! be tested without a network — the plan, the layout, the stamp, the
//! percentage — already are.

use crate::layout::Layout;
use crate::plan::RuntimePlan;
use crate::proc::hidden_command;
use crate::progress::{self, dir_size, pct};
use crate::stamp::{self, Stamp};
use sha2::{Digest, Sha256};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

pub struct Report {
    /// Printed by `--bootstrap-only` so the CI job can calibrate the progress
    /// constants in plan.rs against a real install.
    // Nothing in the GUI path reads it; Task 11's cli.rs is what prints it.
    #[allow(dead_code)]
    pub site_packages_bytes: u64,
}

#[derive(Debug)]
pub struct BootstrapError {
    pub title: String,
    pub message: String,
    pub diagnostics: String,
    pub retryable: bool,
}

impl BootstrapError {
    fn new(title: &str, message: impl Into<String>, retryable: bool) -> Self {
        Self {
            title: title.into(),
            message: message.into(),
            diagnostics: String::new(),
            retryable,
        }
    }

    fn with_diagnostics(mut self, text: impl Into<String>) -> Self {
        self.diagnostics = text.into();
        self
    }
}

/// A `progress(phase, pct, detail)` callback.
///
/// An `Arc` rather than a borrowed trait object because the pip sampler runs
/// on its own thread — a `&dyn` could not cross that boundary.
///
/// `pct == u8::MAX` means "this is subtext only, leave the bar where it is".
/// pip's output arrives far more often than the bar moves, and pinning the two
/// together would make the bar stutter.
pub type OnProgress = Arc<dyn Fn(&'static str, u8, String) + Send + Sync>;

/// One agent for every request the shell makes.
///
/// `timeout_global(None)` is deliberate: ureq's default global timeout would
/// abort a 111 MB download on a slow link. The per-phase connect timeout is
/// what catches a machine with no network, and it is short.
pub fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(None)
        .timeout_connect(Some(Duration::from_secs(20)))
        .user_agent(concat!("local-img/", env!("CARGO_PKG_VERSION")))
        .build()
        .into()
}

pub fn run(
    plan: &RuntimePlan,
    layout: &Layout,
    requirements_path: &Path,
    on_progress: &OnProgress,
) -> Result<Report, BootstrapError> {
    let requirements = std::fs::read(requirements_path).map_err(|e| {
        BootstrapError::new(
            "The app is missing a file it ships with",
            format!("could not read {}: {e}", requirements_path.display()),
            false,
        )
    })?;

    check_disk(plan, layout)?;

    clear_for_rebuild(layout)?;
    std::fs::create_dir_all(&layout.runtime_dir).map_err(|e| {
        BootstrapError::new(
            "Could not create the app's folder",
            format!("{}: {e}", layout.runtime_dir.display()),
            true,
        )
    })?;

    on_progress("python", 0, "downloading the Python engine".into());
    download(plan, &layout.archive_path, on_progress)?;
    verify(&layout.archive_path, plan.python_sha256)?;

    on_progress("python", 99, "unpacking".into());
    extract(&layout.archive_path, &layout.runtime_dir)?;
    let _ = std::fs::remove_file(&layout.archive_path);

    if !layout.interpreter.is_file() {
        return Err(BootstrapError::new(
            "The Python engine did not unpack correctly",
            format!("expected an interpreter at {}", layout.interpreter.display()),
            true,
        ));
    }
    on_progress("python", 100, "engine ready".into());

    install_dependencies(plan, layout, requirements_path, on_progress)?;

    // Last, and atomically. Everything above can be interrupted; only reaching
    // this line means the next launch may skip all of it.
    let expected = Stamp::expected(plan, &requirements);
    stamp::write_atomic(&layout.stamp_path, &expected).map_err(|e| {
        BootstrapError::new(
            "Could not record the finished install",
            format!("{}: {e}", layout.stamp_path.display()),
            true,
        )
    })?;

    Ok(Report {
        site_packages_bytes: dir_size(&layout.site_packages),
    })
}

/// Clears the ground for a rebuild: the stamp first, then the runtime tree.
///
/// The stamp goes FIRST, and separately, because it lives beside the runtime
/// directory rather than inside it — wiping the directory does not touch it.
/// A stamp that survives into a rebuild can certify a runtime whose dependency
/// install was then interrupted: the interpreter is back on disk, the stamp
/// still matches, and the next launch skips the whole bootstrap and starts
/// against a runtime with no torch in it. Removing it here means any
/// interruption from this point on leaves no stamp, which already forces a
/// redo.
///
/// The directory removal's failure is surfaced rather than swallowed: on
/// Windows a still-running python.exe, or a permission problem, would
/// otherwise let `unpack` silently merge into a half-old tree. Absent is the
/// one acceptable outcome.
fn clear_for_rebuild(layout: &Layout) -> Result<(), BootstrapError> {
    let _ = std::fs::remove_file(&layout.stamp_path);

    if let Err(e) = std::fs::remove_dir_all(&layout.runtime_dir) {
        if e.kind() != std::io::ErrorKind::NotFound {
            return Err(BootstrapError::new(
                "Could not clear the previous engine",
                format!(
                    "{}: {e}. If local-img is already running, close it and try again.",
                    layout.runtime_dir.display()
                ),
                true,
            ));
        }
    }
    Ok(())
}

fn check_disk(plan: &RuntimePlan, layout: &Layout) -> Result<(), BootstrapError> {
    // Checked before anything transfers, the way disk_shortfall() already does
    // for model weights — filling the disk and then failing is the worst
    // available outcome.
    let probe = layout
        .data_dir
        .ancestors()
        .find(|p| p.exists())
        .unwrap_or(Path::new("/"));
    let Some(free) = progress::free_disk_bytes(probe) else {
        return Ok(()); // unknowable is not the same as insufficient
    };
    let need = progress::required_disk_bytes(plan);
    if free >= need {
        return Ok(());
    }
    Err(BootstrapError::new(
        "Not enough disk space",
        format!(
            "setting up the engine needs about {} and only {} is free on this drive",
            progress::human_gb(need),
            progress::human_gb(free)
        ),
        true,
    ))
}

fn download(
    plan: &RuntimePlan,
    dest: &Path,
    on_progress: &OnProgress,
) -> Result<(), BootstrapError> {
    let mut response = agent().get(&plan.python_url).call().map_err(|e| {
        BootstrapError::new(
            "Could not reach the download server",
            format!("{} — {e}", plan.python_url),
            true,
        )
    })?;

    // Content-Length survives GitHub's redirect to its asset host, but the
    // pinned size is the fallback if it ever stops doing so.
    let total = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(plan.python_bytes);

    let mut file = std::fs::File::create(dest).map_err(|e| {
        BootstrapError::new("Could not write to disk", format!("{}: {e}", dest.display()), true)
    })?;
    let mut reader = response.body_mut().as_reader();
    let mut buffer = vec![0u8; 256 * 1024];
    let mut done: u64 = 0;
    let mut last_reported = 0u8;

    loop {
        let read = reader.read(&mut buffer).map_err(|e| {
            BootstrapError::new(
                "The download stopped partway",
                format!("after {} — {e}", progress::human_size(done)),
                true,
            )
        })?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read]).map_err(|e| {
            BootstrapError::new("Could not write to disk", format!("{e}"), true)
        })?;
        done += read as u64;
        // Byte-accurate here — Content-Length is known — but still throttled
        // to whole percents so the webview is not woken 400 times a second.
        let now = pct(done, total);
        if now != last_reported {
            last_reported = now;
            on_progress("python", now, format!(
                "{} of {}", progress::human_size(done), progress::human_size(total)
            ));
        }
    }
    file.sync_all().ok();
    Ok(())
}

fn verify(archive: &Path, expected: &str) -> Result<(), BootstrapError> {
    let mut file = std::fs::File::open(archive).map_err(|e| {
        BootstrapError::new("The download vanished", format!("{e}"), true)
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 256 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|e| BootstrapError::new("Could not read the download", format!("{e}"), true))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let actual: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if actual == expected {
        return Ok(());
    }
    // Delete it: leaving a file that fails verification invites a later
    // "resume" that resumes something untrustworthy.
    let _ = std::fs::remove_file(archive);
    Err(BootstrapError::new(
        "The download did not match its checksum",
        format!("expected {expected}, got {actual}. The file was discarded."),
        true,
    ))
}

fn extract(archive: &Path, into: &Path) -> Result<(), BootstrapError> {
    let file = std::fs::File::open(archive)
        .map_err(|e| BootstrapError::new("Could not open the download", format!("{e}"), true))?;
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(file));
    // Preserves the executable bit on bin/python3, which is the difference
    // between a working interpreter and a permission denied.
    tar.set_preserve_permissions(true);
    tar.unpack(into).map_err(|e| {
        BootstrapError::new(
            "Could not unpack the Python engine",
            format!("{}: {e}", into.display()),
            true,
        )
    })
}

fn install_dependencies(
    plan: &RuntimePlan,
    layout: &Layout,
    requirements_path: &Path,
    on_progress: &OnProgress,
) -> Result<(), BootstrapError> {
    on_progress("deps", 0, "preparing".into());

    let base = |args: &[&str]| {
        let mut command = hidden_command(&layout.interpreter);
        command.arg("-m").arg("pip").args(args);
        command
            .arg("--disable-pip-version-check")
            .arg("--no-input")
            .arg("--no-warn-script-location");
        command
    };

    run_pip(base(&["install", "--upgrade", "pip", "wheel"]), plan, layout, on_progress)?;

    // torch first, from its own index. That index serves only torch-family
    // packages, so --index-url alone could not resolve diffusers or fastapi,
    // and --extra-index-url would leave which index wins ambiguous. Installed
    // first, the `torch>=2.4` line in requirements.txt is already satisfied
    // and pip leaves it alone.
    if let Some(index) = plan.torch_index {
        run_pip(
            base(&["install", "--index-url", index, "torch", "torchvision"]),
            plan,
            layout,
            on_progress,
        )?;
    }

    run_pip(
        base(&["install", "-r", &requirements_path.to_string_lossy()]),
        plan,
        layout,
        on_progress,
    )?;

    on_progress("deps", 100, "dependencies ready".into());
    Ok(())
}

fn run_pip(
    mut command: Command,
    plan: &RuntimePlan,
    layout: &Layout,
    on_progress: &OnProgress,
) -> Result<(), BootstrapError> {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // pip inherits the shell's environment otherwise, and a developer's
        // PIP_INDEX_URL or PYTHONPATH would silently redirect the install.
        .env_remove("PYTHONPATH")
        .env_remove("PYTHONHOME")
        .env_remove("PIP_INDEX_URL")
        .env_remove("PIP_EXTRA_INDEX_URL")
        .spawn()
        .map_err(|e| {
            BootstrapError::new(
                "Could not start the installer",
                format!("{}: {e}", layout.interpreter.display()),
                true,
            )
        })?;

    // The bar is sampled from the size of site-packages, not from pip's
    // output. Per-file counters lie — pip resolves, downloads and unpacks in
    // an order it does not announce — but bytes landing in a directory are
    // bytes landing in a directory. The same technique _watch_download() in
    // app.py already uses on Hugging Face blobs, for the same reason.
    let finished = Arc::new(AtomicBool::new(false));
    let sampler = {
        let finished = Arc::clone(&finished);
        let site_packages = layout.site_packages.clone();
        let expected = plan.expected_site_packages;
        let report = Arc::clone(on_progress);
        std::thread::spawn(move || {
            let mut last = 0u8;
            while !finished.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(700));
                if finished.load(Ordering::Relaxed) {
                    break;
                }
                let now = pct(dir_size(&site_packages), expected);
                // Only on change. pip goes silent for minutes while a 2 GB
                // wheel unpacks, and during that stretch this is the only
                // thing that can tell the user anything is still happening.
                if now != last {
                    last = now;
                    report("deps", now, String::new());
                }
            }
        })
    };

    // pip's own line goes underneath the bar as subtext, and its whole output
    // is kept for the diagnostics blob if this fails.
    let stdout = child.stdout.take().expect("piped");
    let stderr = child.stderr.take().expect("piped");
    let transcript = Arc::new(std::sync::Mutex::new(String::new()));

    let stderr_transcript = Arc::clone(&transcript);
    let stderr_reader = std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if let Ok(mut t) = stderr_transcript.lock() {
                t.push_str(&line);
                t.push('\n');
            }
        }
    });

    for line in BufReader::new(stdout).lines().map_while(Result::ok) {
        if let Ok(mut t) = transcript.lock() {
            t.push_str(&line);
            t.push('\n');
        }
        // u8::MAX: subtext only. The sampler owns the bar.
        on_progress("deps", u8::MAX, line.trim().chars().take(110).collect());
    }

    let status = child.wait();
    finished.store(true, Ordering::Relaxed);
    let _ = sampler.join();
    let _ = stderr_reader.join();

    let text = transcript.lock().map(|t| t.clone()).unwrap_or_default();
    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => Err(BootstrapError::new(
            "Installing the dependencies failed",
            // Verbatim, per the spec. A paraphrase of a pip failure helps
            // nobody; the last lines are where the reason actually is.
            format!("pip exited with {s}\n{}", last_lines(&text, 12)),
            true,
        )
        .with_diagnostics(text)),
        Err(e) => Err(BootstrapError::new(
            "The installer stopped unexpectedly",
            format!("{e}"),
            true,
        )
        .with_diagnostics(text)),
    }
}

pub fn last_lines(text: &str, count: usize) -> String {
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    lines[lines.len().saturating_sub(count)..].join("\n")
}

/// Throw away a runtime so the next `run` rebuilds it from scratch.
///
/// Backs the "Reinstall the engine" action. The stamp goes first: interrupted
/// halfway, what is left is a runtime with no stamp, which is exactly the
/// state that already forces a bootstrap.
pub fn discard(layout: &Layout) {
    let _ = std::fs::remove_file(&layout.stamp_path);
    let _ = std::fs::remove_dir_all(&layout.runtime_dir);
    let _ = std::fs::remove_file(&layout.archive_path);
}
