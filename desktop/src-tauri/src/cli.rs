//! The headless bootstrap, for CI.
//!
//! Runs before Tauri is constructed. That is not a convenience: a Linux runner
//! has no display, and building a webview there would fail long before the
//! bootstrap — the thing actually under test — got a chance to run.
//!
//!     local-img-desktop --bootstrap-only --resources <repo> --data-dir <dir>
//!
//! Both directories are required. Defaulting either one would let CI bootstrap
//! against the wrong tree and report a pass that means nothing.

use std::path::PathBuf;
use std::sync::Arc;

pub struct Args {
    pub resources: PathBuf,
    pub data_dir: PathBuf,
}

pub fn parse(argv: &[String]) -> Result<Args, String> {
    let value = |flag: &str| -> Option<&String> {
        argv.iter().position(|a| a == flag).and_then(|i| argv.get(i + 1))
    };
    let resources = value("--resources").ok_or("--resources <dir> is required")?;
    let data_dir = value("--data-dir").ok_or("--data-dir <dir> is required")?;
    Ok(Args {
        resources: PathBuf::from(resources),
        data_dir: PathBuf::from(data_dir),
    })
}

pub fn bootstrap_only() -> i32 {
    let argv: Vec<String> = std::env::args().collect();
    let args = match parse(&argv) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            return 2;
        }
    };

    // `server::start` both `current_dir`s into `resources` and builds its
    // argv as `resources.join("app.py")`. A relative --resources (exactly
    // what CI passes: "../..") would then be applied twice — once by the
    // chdir, once again when the child resolves its own argv against its
    // *new* cwd — landing two directories short of app.py. Canonicalizing
    // once here, before anything downstream sees it, is the fix that does
    // not touch server.rs's frozen contract.
    let resources = match args.resources.canonicalize() {
        Ok(path) => path,
        Err(e) => {
            eprintln!("--resources {}: {e}", args.resources.display());
            return 2;
        }
    };

    let plan = match crate::plan::current() {
        Ok(plan) => plan,
        Err(reason) => {
            eprintln!("unsupported platform: {reason}");
            return 1;
        }
    };
    println!("plan: {} / index {}", plan.triple, plan.index_label());

    let layout = crate::layout::Layout::new(&args.data_dir);
    let requirements = resources.join("requirements.txt");
    let reporter: crate::bootstrap::OnProgress = Arc::new(|phase, pct, detail| {
        // A CI log is a transcript, not a bar: only whole-percent changes and
        // only for the phase, or a 40-minute pip run buries everything else.
        if pct != u8::MAX && pct % 10 == 0 {
            println!("[{phase}] {pct}% {detail}");
        }
    });

    let report = match crate::bootstrap::run(&plan, &layout, &requirements, &reporter) {
        Ok(report) => report,
        Err(e) => {
            eprintln!("bootstrap failed: {} — {}", e.title, e.message);
            eprintln!("{}", e.diagnostics);
            return 1;
        }
    };

    // The number the progress constants in plan.rs are calibrated against.
    println!(
        "MEASURED site-packages: {} bytes ({}); plan expected {} bytes",
        report.site_packages_bytes,
        crate::progress::human_gb(report.site_packages_bytes),
        plan.expected_site_packages
    );

    let outputs = args.data_dir.join("outputs");
    let mut server = match crate::server::start(&layout, &resources, &outputs) {
        Ok(server) => server,
        Err(e) => {
            eprintln!("server failed: {} \n{}", e.message, e.diagnostics);
            return 1;
        }
    };
    println!("server answered on port {}", server.port);

    let code = check_routes(&server);
    let _ = server.is_alive();
    server.shutdown();
    code
}

fn check_routes(server: &crate::server::Server) -> i32 {
    let client = crate::bootstrap::agent();
    for route in ["/api/health", "/api/models", "/api/busy"] {
        let url = format!("http://127.0.0.1:{}{route}", server.port);
        match client
            .get(&url)
            .header(crate::server::TOKEN_HEADER, &server.token)
            .call()
        {
            Ok(mut response) => {
                let body = response.body_mut().read_to_string().unwrap_or_default();
                println!("{route}: ok ({} bytes)", body.len());
            }
            Err(e) => {
                eprintln!("{route}: FAILED — {e}");
                eprintln!("{}", server.tail());
                return 1;
            }
        }
    }
    // The gate has to be live, not merely present: without a token the same
    // routes must be refused. Two probes, because they fail for different
    // reasons — no credential at all, and the token in the channel `/` alone
    // accepts. If the second one succeeds, the query parameter is being
    // honoured API-wide again and the header was pointless.
    for (label, url) in [
        ("no credential", format!("http://127.0.0.1:{}/api/models", server.port)),
        (
            "query token off /",
            format!("http://127.0.0.1:{}/api/models?token={}", server.port, server.token),
        ),
    ] {
        match client.get(&url).call() {
            Err(_) => println!("gate: ok ({label} is refused)"),
            Ok(r) if r.status() == 403 => println!("gate: ok ({label} gets 403)"),
            Ok(r) => {
                eprintln!("gate: FAILED — {label} got {}", r.status());
                return 1;
            }
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn both_directories_are_read_from_the_arguments() {
        let parsed = parse(&args(&[
            "local-img-desktop", "--bootstrap-only",
            "--resources", "/repo", "--data-dir", "/tmp/data",
        ]))
        .unwrap();
        assert_eq!(parsed.resources, std::path::Path::new("/repo"));
        assert_eq!(parsed.data_dir, std::path::Path::new("/tmp/data"));
    }

    #[test]
    fn order_does_not_matter() {
        let parsed = parse(&args(&[
            "x", "--data-dir", "/tmp/data", "--bootstrap-only", "--resources", "/repo",
        ]))
        .unwrap();
        assert_eq!(parsed.resources, std::path::Path::new("/repo"));
    }

    #[test]
    fn a_missing_directory_is_an_error_rather_than_a_default() {
        // Defaulting would let CI silently bootstrap against the wrong tree
        // and report a pass that means nothing.
        assert!(parse(&args(&["x", "--bootstrap-only"])).is_err());
        assert!(parse(&args(&["x", "--bootstrap-only", "--resources", "/repo"])).is_err());
        assert!(parse(&args(&["x", "--bootstrap-only", "--resources"])).is_err());
    }
}
