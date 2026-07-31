//! What the bootstrap needs to know before it touches the network.
//!
//! `plan_for` is pure over `(os, arch, has_nvidia)` because the interesting
//! cases are the ones this project's hardware cannot produce: a Windows box
//! with an NVIDIA card, a Linux box without one, an Intel Mac that has to be
//! turned away rather than half-supported.
//!
//! The checksums are pinned here in the source rather than fetched next to the
//! archive. A checksum served from the same place as the file it describes
//! proves only that the download was not corrupted in transit.

use crate::proc::hidden_program;
use std::process::Stdio;

/// The python-build-standalone release. Bumping it means re-pinning all three
/// checksums from that release's SHA256SUMS.
pub const PBS_TAG: &str = "20260728";
pub const PYTHON_VERSION: &str = "3.12.13";
pub const PYTHON_MINOR: &str = "3.12";

/// cu126 rather than a newer index: it carries current torch and asks less of
/// the installed NVIDIA driver than cu130 does. cu124 is frozen at torch 2.6.
pub const CUDA_INDEX: &str = "https://download.pytorch.org/whl/cu126";
pub const CPU_INDEX: &str = "https://download.pytorch.org/whl/cpu";

/// Expected `site-packages` totals, in bytes, used only to turn a growing
/// directory into a percentage. Wrong by 20% costs a bar that moves unevenly;
/// wrong by an order of magnitude costs a bar that looks stuck.
/// `--bootstrap-only` (see .github/workflows/smoke.yml) prints the measured
/// size on whatever platform it runs on, so each of these gets replaced with
/// a real number as that platform is measured — macOS below is; Windows and
/// Linux are not yet.
///
/// Measured by `--bootstrap-only` on an M-series Mac: 922,053,753 bytes of
/// site-packages after torch and requirements.txt install from PyPI. Rounded
/// up, because a bar that reaches 99 slightly late reads better than one that
/// sits there. Wrong by 20% costs a bar that moves unevenly; wrong by an order
/// of magnitude costs a bar that looks stuck.
const SITE_PACKAGES_MACOS: u64 = 925_000_000;
/// Not yet measured: only the CI smoke job's Windows and Ubuntu jobs can
/// measure this, and neither has run yet. Take the larger of the two the
/// first time they do.
const SITE_PACKAGES_CPU: u64 = 1_400_000_000;
/// Not measured: the cuda-index job is manual and has not been run. This is
/// still the estimate. Replace it the first time that job runs.
const SITE_PACKAGES_CUDA: u64 = 7_000_000_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimePlan {
    pub triple: &'static str,
    pub python_url: String,
    pub python_sha256: &'static str,
    /// The archive's exact size. Used for the progress bar when a redirect
    /// drops Content-Length, and to tell the user up front what this costs.
    pub python_bytes: u64,
    /// None means plain PyPI — which on Apple Silicon is the right answer,
    /// because MPS support is built into the published wheel.
    pub torch_index: Option<&'static str>,
    pub expected_site_packages: u64,
}

impl RuntimePlan {
    /// How the stamp records the index choice, so switching from CPU to CUDA
    /// wheels forces a reinstall.
    pub fn index_label(&self) -> &'static str {
        self.torch_index.unwrap_or("pypi")
    }

    /// Everything the first download costs, for the welcome screen.
    pub fn total_download_bytes(&self) -> u64 {
        self.python_bytes + self.expected_site_packages
    }
}

fn archive(triple: &'static str, sha: &'static str, bytes: u64) -> (&'static str, &'static str, u64) {
    (triple, sha, bytes)
}

/// The plan for a platform, or a sentence explaining why there isn't one.
pub fn plan_for(os: &str, arch: &str, has_nvidia: bool) -> Result<RuntimePlan, String> {
    let (triple, sha, bytes) = match (os, arch) {
        ("macos", "aarch64") => archive(
            "aarch64-apple-darwin",
            "12d6700f7e8f222639f0ee5bbd173082c3041aeb65af8f9828e4216bc8047de6",
            25_149_265,
        ),
        ("windows", "x86_64") => archive(
            "x86_64-pc-windows-msvc",
            "8a0e1ded37e11f4c72b9671bf134bb478b1b2d55efe53a3d6e589b166f1bf2e1",
            46_148_055,
        ),
        ("linux", "x86_64") => archive(
            "x86_64-unknown-linux-gnu",
            "fd9d70e1e1ed3f6caccb4e2eefe570aa07589c8f86ddf0e87f68a96cd14272e1",
            111_358_187,
        ),
        ("macos", _) => {
            return Err(
                "local-img needs an Apple Silicon Mac. On an Intel Mac there is no \
                 Metal backend for PyTorch, so every image would be generated on the \
                 CPU — minutes each, and around 14 GB of RAM for the larger models. \
                 Shipping that would disappoint rather than help."
                    .into(),
            )
        }
        ("windows", _) | ("linux", _) => {
            return Err(format!(
                "local-img needs a 64-bit Intel or AMD processor on {os}; this machine \
                 reports {arch}. PyTorch publishes no wheels for it."
            ))
        }
        _ => {
            return Err(format!(
                "local-img has no build for {os}. It supports macOS on Apple Silicon, \
                 and Windows and Linux on x86-64."
            ))
        }
    };

    // Literally what the spec asks for: ask nvidia-smi first, and only fall
    // back to the platform default when it does not answer.
    let torch_index = if has_nvidia {
        Some(CUDA_INDEX)
    } else if os == "macos" {
        None
    } else {
        Some(CPU_INDEX)
    };

    let expected_site_packages = match torch_index {
        Some(CUDA_INDEX) => SITE_PACKAGES_CUDA,
        Some(_) => SITE_PACKAGES_CPU,
        None => SITE_PACKAGES_MACOS,
    };

    Ok(RuntimePlan {
        triple,
        python_url: format!(
            "https://github.com/astral-sh/python-build-standalone/releases/download/\
             {PBS_TAG}/cpython-{PYTHON_VERSION}+{PBS_TAG}-{triple}-install_only.tar.gz"
        ),
        python_sha256: sha,
        python_bytes: bytes,
        torch_index,
        expected_site_packages,
    })
}

/// Whether an NVIDIA driver is installed and answering.
///
/// `nvidia-smi -L` rather than a bare invocation: it lists the cards and exits
/// immediately, where the bare form prints a full table and, on a machine with
/// a driver but no card, can hang for seconds.
pub fn nvidia_present() -> bool {
    hidden_program("nvidia-smi")
        .arg("-L")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// The plan for the machine this is running on.
pub fn current() -> Result<RuntimePlan, String> {
    plan_for(
        std::env::consts::OS,
        std::env::consts::ARCH,
        nvidia_present(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apple_silicon_uses_pypi_because_mps_is_built_in() {
        let plan = plan_for("macos", "aarch64", false).unwrap();
        assert!(plan.python_url.contains("aarch64-apple-darwin"));
        assert_eq!(plan.torch_index, None);
        assert_eq!(plan.python_bytes, 25_149_265);
    }

    #[test]
    fn nvidia_anywhere_selects_the_cuda_index() {
        for os in ["windows", "linux"] {
            let plan = plan_for(os, "x86_64", true).unwrap();
            assert_eq!(plan.torch_index, Some(CUDA_INDEX));
            assert!(plan.expected_site_packages > 5_000_000_000,
                    "{os}: cuda wheels are the large case");
        }
    }

    #[test]
    fn no_nvidia_off_mac_selects_the_cpu_index() {
        // Generating on CPU is impractical, but pulling 7 GB of CUDA wheels
        // onto a machine with no NVIDIA card is worse.
        for os in ["windows", "linux"] {
            let plan = plan_for(os, "x86_64", false).unwrap();
            assert_eq!(plan.torch_index, Some(CPU_INDEX));
            assert!(plan.expected_site_packages < 3_000_000_000);
        }
    }

    #[test]
    fn each_platform_gets_its_own_archive() {
        let mac = plan_for("macos", "aarch64", false).unwrap();
        let win = plan_for("windows", "x86_64", false).unwrap();
        let linux = plan_for("linux", "x86_64", false).unwrap();
        assert!(win.python_url.contains("x86_64-pc-windows-msvc"));
        assert!(linux.python_url.contains("x86_64-unknown-linux-gnu"));
        let urls = [&mac.python_url, &win.python_url, &linux.python_url];
        for (i, a) in urls.iter().enumerate() {
            for b in urls.iter().skip(i + 1) {
                assert_ne!(a, b, "two platforms share an archive");
            }
        }
    }

    #[test]
    fn every_checksum_is_a_real_sha256() {
        for (os, arch) in [("macos", "aarch64"), ("windows", "x86_64"), ("linux", "x86_64")] {
            let plan = plan_for(os, arch, false).unwrap();
            assert_eq!(plan.python_sha256.len(), 64, "{os}");
            assert!(
                plan.python_sha256.chars().all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()),
                "{os}: checksums are lowercase hex"
            );
        }
    }

    #[test]
    fn every_url_points_at_the_pinned_release() {
        for (os, arch) in [("macos", "aarch64"), ("windows", "x86_64"), ("linux", "x86_64")] {
            let plan = plan_for(os, arch, false).unwrap();
            assert!(plan.python_url.starts_with("https://github.com/astral-sh/"));
            assert!(plan.python_url.contains(PBS_TAG), "{os}: unpinned release");
            assert!(plan.python_url.contains(PYTHON_VERSION), "{os}: wrong python");
            assert!(plan.python_url.ends_with("-install_only.tar.gz"),
                    "{os}: one archive format for all three platforms");
        }
    }

    #[test]
    fn unsupported_hardware_is_named_rather_than_attempted() {
        // Each of these would otherwise fail deep inside pip with a message
        // about wheels, which tells the user nothing they can act on.
        let intel_mac = plan_for("macos", "x86_64", false).unwrap_err();
        assert!(intel_mac.contains("Apple Silicon"), "got: {intel_mac}");

        let arm_windows = plan_for("windows", "aarch64", false).unwrap_err();
        assert!(arm_windows.contains("64-bit"), "got: {arm_windows}");

        let arm_linux = plan_for("linux", "aarch64", false).unwrap_err();
        assert!(arm_linux.contains("64-bit"), "got: {arm_linux}");

        assert!(plan_for("freebsd", "x86_64", false).is_err());
    }

    #[test]
    fn this_machine_has_a_plan() {
        // The one case that exercises the real std::env::consts values.
        let plan = current().expect("the developer's machine must be supported");
        assert!(!plan.python_url.is_empty());
    }

    #[test]
    fn nvidia_detection_does_not_panic_without_a_card() {
        // The developer's Mac has none; CI runners have none either. What is
        // being asserted is that a missing binary is a `false`, not an error.
        let _ = nvidia_present();
    }
}
