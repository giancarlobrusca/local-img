"""Where each thing lives.

Two environment variables and nothing else. There is no platform logic here on
purpose — no knowledge of macOS bundles, Windows known folders, or XDG. The
desktop shell resolves those and passes the answers in; with both variables
unset every path is exactly what the repo flow has always used, which is what
lets `./setup.sh` + `./run.sh` keep working unchanged.

    LOCAL_IMG_DATA_DIR   the hardware profile (and, for the shell, the runtime
                         and logs, which Python never touches)
    LOCAL_IMG_OUTPUTS    generated PNGs and their sidecars

Both are read at call time, not at import, so a test can set one and see the
effect without reloading modules.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).parent


def _from_env(name: str) -> Path | None:
    """The variable as a path, or None when unset, blank, or all whitespace."""
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser() if raw else None


def data_dir() -> Path:
    """Where the hardware profile lives. Repo mode keeps ./.local-img/."""
    return _from_env("LOCAL_IMG_DATA_DIR") or (ROOT / ".local-img")


def profile_path() -> Path:
    return data_dir() / "profile.json"


def outputs_dir() -> Path:
    """Where renders land. Repo mode keeps ./outputs/."""
    return _from_env("LOCAL_IMG_OUTPUTS") or (ROOT / "outputs")


def hf_cache_dir() -> Path:
    """The Hugging Face cache — the same place in both modes, deliberately.

    Someone who already ran the repo flow does not re-download 7 GB of weights
    when they install the app, and download.py needs no change at all.
    """
    return Path.home() / ".cache" / "huggingface"


def disk_probe_dir() -> Path:
    """The directory whose free space decides whether a download fits.

    Weights land in the Hugging Face cache under the home directory in both
    modes. The repo — or, installed, the app bundle in /Applications or
    Program Files — may sit on an entirely different volume, so measuring free
    space next to the code answers the wrong question. Home is the right volume
    and, unlike hf_cache_dir(), it exists before the first download.
    """
    return Path.home()
