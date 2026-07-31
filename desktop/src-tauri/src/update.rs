//! Checking GitHub for a newer release.
//!
//! There is no auto-update: the app says a newer version exists and links to
//! it. Failing silently is the correct behaviour with no network — an app that
//! cannot reach GitHub still generates images perfectly well offline, which is
//! the whole point of it.

use crate::events::UpdateInfo;

const RELEASES_API: &str =
    "https://api.github.com/repos/giancarlobrusca/local-img/releases/latest";

fn parts(version: &str) -> Option<Vec<u32>> {
    let trimmed = version.trim().trim_start_matches('v');
    if trimmed.is_empty() {
        return None;
    }
    trimmed.split('.').map(|p| p.parse::<u32>().ok()).collect()
}

/// Whether `candidate` is a later release than `current`.
///
/// Anything that does not parse as dotted numbers answers false. A banner is
/// an interruption, and interrupting on a tag nobody can interpret is worse
/// than missing an update.
pub fn is_newer(candidate: &str, current: &str) -> bool {
    let (Some(a), Some(b)) = (parts(candidate), parts(current)) else {
        return false;
    };
    let len = a.len().max(b.len());
    for i in 0..len {
        // Missing components are zero: 0.2 is later than 0.1.9, and 0.1 is
        // the same release as 0.1.0.
        let (x, y) = (a.get(i).copied().unwrap_or(0), b.get(i).copied().unwrap_or(0));
        if x != y {
            return x > y;
        }
    }
    false
}

pub fn check(current_version: &str) -> Option<UpdateInfo> {
    // The bounded agent, not bootstrap's. bootstrap::agent() sets
    // timeout_global(None) so a 111 MB download is never cut off mid-transfer;
    // for a 5 KB JSON call that is the wrong guarantee. A captive portal that
    // accepts the connection and never answers would park this thread — and
    // the WebviewWindow clone it captured — for the life of the process.
    let body = crate::server::poll_agent()
        .get(RELEASES_API)
        .header("Accept", "application/vnd.github+json")
        .call()
        .ok()?
        .body_mut()
        .read_to_string()
        .ok()?;
    let release: serde_json::Value = serde_json::from_str(&body).ok()?;
    let tag = release["tag_name"].as_str()?;
    if !is_newer(tag, current_version) {
        return None;
    }
    Some(UpdateInfo {
        version: tag.to_string(),
        url: release["html_url"]
            .as_str()
            .unwrap_or("https://github.com/giancarlobrusca/local-img/releases/latest")
            .to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_higher_version_is_newer() {
        assert!(is_newer("v0.2.0", "0.1.0"));
        assert!(is_newer("0.1.1", "0.1.0"));
        assert!(is_newer("1.0.0", "0.99.99"));
        assert!(is_newer("v0.10.0", "v0.9.0"), "ten is not less than nine");
    }

    #[test]
    fn the_same_or_older_version_is_not_newer() {
        assert!(!is_newer("0.1.0", "0.1.0"));
        assert!(!is_newer("v0.1.0", "0.1.0"), "the v prefix is not a difference");
        assert!(!is_newer("0.0.9", "0.1.0"));
    }

    #[test]
    fn a_shorter_version_compares_as_zero_padded() {
        assert!(is_newer("0.2", "0.1.9"));
        assert!(!is_newer("0.1", "0.1.0"));
    }

    #[test]
    fn unparseable_tags_never_claim_to_be_newer() {
        // A banner is an interruption. Anything ambiguous stays quiet.
        assert!(!is_newer("nightly", "0.1.0"));
        assert!(!is_newer("", "0.1.0"));
        assert!(!is_newer("v1.x", "0.1.0"));
    }
}
