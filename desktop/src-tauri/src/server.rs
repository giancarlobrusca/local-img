//! Running `app.py` as a child, and cleaning up after it.
//!
//! Two directions of orphaning are possible and both are closed. Window closed
//! → the shell terminates the child here. Shell killed abruptly → nothing here
//! gets to run, which is why app.py has its own parent watchdog.

use crate::bootstrap::last_lines;
use crate::layout::Layout;
use crate::proc::hidden_command;
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How long the child gets to import torch and bind its port before the shell
/// gives up. Loading torch on a cold filesystem cache is genuinely slow.
const HEALTH_TIMEOUT: Duration = Duration::from_secs(120);
const LOG_MAX_BYTES: u64 = 2_000_000;
const TAIL_LINES: usize = 60;
/// Must match `app.TOKEN_HEADER`. The shell authenticates with a header rather
/// than a query parameter so the token stays out of logs and history.
pub const TOKEN_HEADER: &str = "x-local-img-token";

#[derive(Debug)]
pub struct ServerError {
    pub message: String,
    pub diagnostics: String,
}

/// The last N lines of the child's output, for the failure screen.
#[derive(Clone)]
pub struct Tail {
    lines: Arc<Mutex<VecDeque<String>>>,
    limit: usize,
}

impl Tail {
    pub fn new(limit: usize) -> Self {
        Self {
            lines: Arc::new(Mutex::new(VecDeque::with_capacity(limit))),
            limit,
        }
    }

    pub fn push(&self, line: String) {
        if let Ok(mut lines) = self.lines.lock() {
            if lines.len() == self.limit {
                lines.pop_front();
            }
            lines.push_back(line);
        }
    }

    pub fn text(&self) -> String {
        self.lines
            .lock()
            .map(|l| l.iter().cloned().collect::<Vec<_>>().join("\n"))
            .unwrap_or_default()
    }
}

pub struct Server {
    child: Child,
    pub port: u16,
    pub token: String,
    tail: Tail,
}

/// Ask the OS for a port nobody is using.
///
/// Binding port 0 and reading back what was assigned is the only way to get an
/// answer that is true rather than hopeful. A race window remains between the
/// drop here and the child's own bind, which `start` handles by retrying.
pub fn free_port() -> std::io::Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

/// 128 bits of randomness, hex encoded. Fresh on every launch, so a token
/// leaked into browser history is worthless by the next start.
pub fn random_token() -> String {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes).expect("the OS must provide randomness");
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn handoff_url(port: u16, token: &str, firstrun: bool) -> String {
    let suffix = if firstrun { "&firstrun=1" } else { "" };
    format!("http://127.0.0.1:{port}/?token={token}{suffix}")
}

/// A bounded agent for the two calls that must never hang the UI.
///
/// `bootstrap::agent()` deliberately has no global timeout, because a 111 MB
/// download must not be cut off mid-transfer. These two calls want the
/// opposite guarantee. A child that accepts the connection and then never
/// answers would otherwise block `wait_healthy` indefinitely — its own
/// deadline is only checked *between* attempts, never during one — and would
/// freeze the close-confirmation prompt while the user is trying to quit.
fn poll_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(5)))
        .build()
        .into()
}

/// Set the current log aside if it has grown past `max_bytes`.
///
/// One generation, not many. This exists so a Windows user's bug report is a
/// readable file, not so months of launches are archived.
pub fn rotate_log(path: &Path, max_bytes: u64) {
    let Ok(meta) = std::fs::metadata(path) else {
        return;
    };
    if meta.len() <= max_bytes {
        return;
    }
    let rotated = path.with_extension("1.log");
    let _ = std::fs::remove_file(&rotated);
    let _ = std::fs::rename(path, &rotated);
}

impl Server {
    pub fn url_with_token(&self, firstrun: bool) -> String {
        handoff_url(self.port, &self.token, firstrun)
    }

    pub fn tail(&self) -> String {
        self.tail.text()
    }

    /// Whether a render is in flight, so the shell can ask before closing.
    ///
    /// A download is deliberately not counted: it resumes from the Hugging
    /// Face cache, where a four-minute flux render that has not written its
    /// PNG yet is simply lost.
    pub fn is_generating(&self) -> bool {
        poll_agent()
            .get(&format!("http://127.0.0.1:{}/api/busy", self.port))
            // A header, not a query parameter. app.py accepts `?token=` on `/`
            // alone, so the token never reaches a log, a Referer, or history.
            .header(TOKEN_HEADER, &self.token)
            .call()
            .ok()
            .and_then(|mut r| r.body_mut().read_to_string().ok())
            .and_then(|body| serde_json::from_str::<serde_json::Value>(&body).ok())
            .and_then(|v| v["generating"].as_bool())
            .unwrap_or(false)
    }

    /// Whether the child is still running.
    pub fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    /// Ask, then insist. `kill` is a SIGKILL on Unix and a TerminateProcess on
    /// Windows; neither runs Python's atexit handlers, so the grace period is
    /// what gives uvicorn a chance to close its socket cleanly.
    pub fn shutdown(mut self) {
        let deadline = Instant::now() + Duration::from_secs(5);
        let _ = self.child.kill();
        while Instant::now() < deadline {
            if let Ok(Some(_)) = self.child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let _ = self.child.wait();
    }
}

impl Drop for Server {
    /// The backstop for every path that never reaches `shutdown` — a panic
    /// unwinding out of the caller's setup, or an early return on some later
    /// error. `std::process::Child` neither kills nor reaps on drop, so
    /// without this a live Python process holding a multi-GB model can outlive
    /// the shell that started it, and leave a zombie behind as well.
    ///
    /// Harmless after an explicit `shutdown`: the child is already reaped, so
    /// both calls simply return an error that is discarded.
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Start the server, retrying on a port collision.
///
/// Three attempts: `free_port` closes its listener before the child binds, so
/// something else can take the port in between. It is a small window and it
/// does happen.
pub fn start(
    layout: &Layout,
    resources: &Path,
    outputs: &Path,
) -> Result<Server, ServerError> {
    let mut last: Option<ServerError> = None;
    for _ in 0..3 {
        match spawn_once(layout, resources, outputs) {
            Ok(server) => return Ok(server),
            Err(e) => {
                // Lowercase the haystack, not just the needles. asyncio builds
                // this message from `err.strerror.lower()`, so Windows reports
                // "only one usage of each socket address..." in lower case and
                // a capitalised needle would never match — collisions there
                // would fail on the first attempt instead of retrying. Python
                // prints "[Errno 48]" with a capital E, hence the brackets.
                let lowered = e.diagnostics.to_ascii_lowercase();
                let addr_in_use = lowered.contains("address already in use")
                    || lowered.contains("only one usage of each socket address")
                    || lowered.contains("[errno 48]");
                if !addr_in_use {
                    return Err(e);      // a real failure — report it as-is
                }
                last = Some(e);
            }
        }
    }
    // Three collisions running. Hand back the child's own last words rather
    // than a summary of them: uvicorn's message names the port it tried.
    Err(last.expect("the loop reaches here only after recording an error"))
}

fn spawn_once(
    layout: &Layout,
    resources: &Path,
    outputs: &Path,
) -> Result<Server, ServerError> {
    let port = free_port().map_err(|e| ServerError {
        message: "could not find a free port on this machine".into(),
        diagnostics: format!("{e}"),
    })?;
    let token = random_token();

    std::fs::create_dir_all(&layout.logs_dir).ok();
    let log_path = layout.logs_dir.join("server.log");
    rotate_log(&log_path, LOG_MAX_BYTES);

    let mut child = hidden_command(&layout.interpreter)
        .arg(resources.join("app.py"))
        .current_dir(resources)
        .env("LOCAL_IMG_PORT", port.to_string())
        .env("LOCAL_IMG_TOKEN", &token)
        .env("LOCAL_IMG_DATA_DIR", &layout.data_dir)
        .env("LOCAL_IMG_OUTPUTS", outputs)
        .env("LOCAL_IMG_PARENT_PID", std::process::id().to_string())
        .env("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        .env("HF_XET_HIGH_PERFORMANCE", "1")
        // Line-buffered, so the log has content while the child is running
        // rather than only after it dies — which is exactly when it matters.
        .env("PYTHONUNBUFFERED", "1")
        // Force UTF-8 out of the child. CPython otherwise writes stdout in the
        // locale codepage on Windows (cp1252, cp932), so a traceback naming a
        // path with an accent in it arrives as bytes the tee has to guess at.
        .env("PYTHONIOENCODING", "utf-8")
        // huggingface_hub's tqdm bars write \r without \n, so a multi-GB
        // download becomes one enormous "line": it bloats the rotated log for
        // the rest of the session and lands in the 60-line tail as a single
        // several-hundred-KB string, crowding out the messages worth reading.
        .env("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        // A developer's PYTHONPATH must not reach into the private runtime.
        .env_remove("PYTHONPATH")
        .env_remove("PYTHONHOME")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| ServerError {
            message: "could not start the generation server".into(),
            diagnostics: format!("{}: {e}", layout.interpreter.display()),
        })?;

    let tail = Tail::new(TAIL_LINES);

    // stdout and stderr are tee'd separately: both into the rotated log, and
    // both into the in-memory tail the failure screen shows. Without this, a
    // Python exception on a Windows machine is a blank window and a bug report
    // that says "it doesn't work".
    if let Some(stdout) = child.stdout.take() {
        tee(stdout, tail.clone(), log_path.clone());
    }
    if let Some(stderr) = child.stderr.take() {
        tee(stderr, tail.clone(), log_path.clone());
    }

    let mut server = Server { child, port, token, tail };

    match wait_healthy(&mut server) {
        Ok(()) => Ok(server),
        Err(message) => {
            let diagnostics = server.tail();
            server.shutdown();
            Err(ServerError { message, diagnostics })
        }
    }
}

fn tee<R: std::io::Read + Send + 'static>(stream: R, tail: Tail, log_path: PathBuf) {
    std::thread::spawn(move || {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .ok();
        // Bytes, not `lines()`. `lines()` yields Err on invalid UTF-8, and
        // ending the loop there would kill this thread permanently — which
        // does far more than lose a log line. A dead tee stops draining the
        // pipe, so once the OS buffer fills (~64 KB) the child blocks forever
        // on its next print and the whole app hangs with no error anywhere.
        // CPython writes stdout in the locale codepage on Windows, so one
        // accented character in a traceback is enough to trigger it. Lossy
        // decoding costs a single glyph instead of the entire channel.
        let mut reader = BufReader::new(stream);
        let mut raw = Vec::new();
        loop {
            raw.clear();
            match reader.read_until(b'\n', &mut raw) {
                Ok(0) | Err(_) => break,     // EOF, or the child is gone
                Ok(_) => {}
            }
            while matches!(raw.last(), Some(b'\n') | Some(b'\r')) {
                raw.pop();
            }
            let text = String::from_utf8_lossy(&raw).into_owned();
            if let Some(file) = file.as_mut() {
                let _ = writeln!(file, "{text}");
            }
            tail.push(text);
        }
    });
}

fn wait_healthy(server: &mut Server) -> Result<(), String> {
    let url = format!("http://127.0.0.1:{}/api/health", server.port);
    let deadline = Instant::now() + HEALTH_TIMEOUT;
    let client = poll_agent();
    while Instant::now() < deadline {
        if !server.is_alive() {
            return Err(format!(
                "the generation server stopped before it finished starting:\n{}",
                last_lines(&server.tail(), 12)
            ));
        }
        if client.get(&url).call().is_ok() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(400));
    }
    Err(format!(
        "the generation server did not answer within {} seconds",
        HEALTH_TIMEOUT.as_secs()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_free_port_is_plausible_and_not_privileged() {
        let first = free_port().unwrap();
        assert!(first >= 1024, "an ephemeral port is never privileged");

        let second = free_port().unwrap();
        assert!(second >= 1024);

        // Deliberately NOT asserted: that `first` can be re-bound right now.
        // free_port() closes its listener before returning, so the port is up
        // for grabs from that instant — under a parallel test run the kernel
        // hands it back out often enough that asserting otherwise flakes
        // (observed failing ~60% of full-suite runs on macOS with
        // AddrInUse/errno 48). That race is real and unavoidable without
        // handing a bound socket to a Python child, which is not possible
        // here. `start()` answers it by retrying on an address collision
        // rather than by pretending it cannot happen — so a test asserting it
        // cannot happen would contradict the design it is meant to protect.
    }

    #[test]
    fn tokens_are_long_random_hex() {
        let a = random_token();
        let b = random_token();
        assert_eq!(a.len(), 32);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, b, "a token is per launch, not per build");
    }

    #[test]
    fn the_handoff_url_carries_the_token_and_the_firstrun_flag() {
        let url = handoff_url(7788, "abc123", true);
        assert_eq!(url, "http://127.0.0.1:7788/?token=abc123&firstrun=1");
        assert_eq!(handoff_url(51234, "abc123", false),
                   "http://127.0.0.1:51234/?token=abc123");
    }

    #[test]
    fn an_oversized_log_is_rotated_once() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("server.log");
        std::fs::write(&log, vec![b'x'; 500]).unwrap();

        rotate_log(&log, 1_000);
        assert!(log.exists(), "under the limit, nothing moves");
        assert!(!dir.path().join("server.1.log").exists());

        rotate_log(&log, 100);
        assert!(!log.exists(), "over the limit, the current log is set aside");
        let rotated = std::fs::read(dir.path().join("server.1.log")).unwrap();
        assert_eq!(rotated.len(), 500);
    }

    #[test]
    fn rotating_replaces_the_previous_rotation_rather_than_accumulating() {
        // Two rotations must leave two files, not three — this is a debugging
        // channel, not an archive.
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("server.log");
        std::fs::write(&log, b"first").unwrap();
        rotate_log(&log, 1);
        std::fs::write(&log, b"second").unwrap();
        rotate_log(&log, 1);
        assert_eq!(std::fs::read(dir.path().join("server.1.log")).unwrap(), b"second");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn rotating_a_log_that_does_not_exist_yet_is_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("absent.log");
        rotate_log(&log, 1);
        // The real claim: it neither creates the file nor invents a rotation.
        assert!(!log.exists());
        assert!(!dir.path().join("absent.1.log").exists());
    }

    #[test]
    fn the_tail_keeps_the_last_lines_and_drops_the_rest() {
        let tail = Tail::new(3);
        for i in 0..10 {
            tail.push(format!("line {i}"));
        }
        assert_eq!(tail.text(), "line 7\nline 8\nline 9");
    }
}
