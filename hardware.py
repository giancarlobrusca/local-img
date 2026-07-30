"""Hardware detection, per-model fit rules, and speed estimates.

Nothing here imports app.py — `detect()` takes the device string as an argument,
so this module stays importable, testable without a server or a GPU, and free of
a circular import. `fit()` and `estimate()` are pure: data in, data out.

Memory is GiB (bytes / 2**30) throughout: total_ram_gb, budget_gb, and the
min_ram_gb / min_budget_gb columns in models.py all use the same unit, so a
16 GB machine reports 16.0 and compares cleanly against a 16 threshold. Disk is
decimal GB (bytes / 1e9), matching download_gb.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).parent

# Bumping this invalidates every stored profile and forces a re-scan.
SCHEMA_VERSION = 1
PROFILE_PATH = ROOT / ".local-img" / "profile.json"

REFERENCE_GPU_CORES = 16   # M1 Pro — perf_factor 1.0
CPU_PERF_FACTOR = 0.04
CPU_BUDGET_FRACTION = 0.6
# app.py runs fp32 on CPU: no fp16 kernels for much of the UNet.
CPU_COST_MULTIPLIER = 2.0


@dataclass
class HardwareProfile:
    schema_version: int
    device: str              # mps | cuda | cpu
    platform: str            # "darwin-arm64"
    chip: str                # "Apple M1 Pro"
    gpu_cores: int | None    # Apple Silicon only
    total_ram_gb: float      # GiB
    budget_gb: float         # GiB actually available for weights + activations
    free_disk_gb: float      # decimal GB
    tier: str                # cpu | low | mid | high | ultra
    perf_factor: float       # speed relative to the reference machine
    partial: bool            # detection ran but some field could not be read
    skipped: bool            # the user declined the analysis — not a measurement
    detected_at: str         # ISO 8601


@dataclass
class Fit:
    fits: bool
    reason: str        # "" when it fits; otherwise a concrete sentence
    max_batch: int
    recommended: bool


# --------------------------------------------------------------------- fit ---

def max_batch_for(budget_gb: float) -> int:
    if budget_gb < 10:
        return 1
    if budget_gb < 20:
        return 2
    return 4


def fit(profile: HardwareProfile | None, spec) -> Fit:
    """Whether `spec` runs on `profile`, and if not, which number blocks it.

    `recommended` is always False here — it depends on the rest of the catalog.
    Use fit_all() to get the recommendation set.
    """
    if profile is None or profile.skipped:
        # Nothing was measured, so nothing may be claimed. Show everything.
        return Fit(fits=True, reason="", max_batch=4, recommended=False)

    batch = max_batch_for(profile.budget_gb)
    multiplier = CPU_COST_MULTIPLIER if profile.device == "cpu" else 1.0
    need_budget = spec.min_budget_gb * multiplier

    if profile.budget_gb < need_budget:
        return Fit(
            fits=False,
            reason=f"needs {need_budget:g} GB of usable memory; "
                   f"this machine has {profile.budget_gb:.1f} GB",
            max_batch=batch,
            recommended=False,
        )
    if profile.total_ram_gb < spec.min_ram_gb:
        return Fit(
            fits=False,
            reason=f"needs {spec.min_ram_gb:g} GB of system RAM; "
                   f"this machine has {profile.total_ram_gb:.0f} GB",
            max_batch=batch,
            recommended=False,
        )
    return Fit(fits=True, reason="", max_batch=batch, recommended=False)


def fit_all(profile: HardwareProfile | None, specs) -> dict[str, Fit]:
    """fit() across a catalog, with exactly one model marked `recommended`.

    The pick is the highest quality_rank among the models that fit, ties broken
    by the lower baseline_seconds. A missing or skipped profile recommends
    nothing: there is no evidence to recommend from.
    """
    fits = {s.key: fit(profile, s) for s in specs}
    if profile is None or profile.skipped:
        return fits
    winners = [s for s in specs if fits[s.key].fits]
    if winners:
        best = max(winners, key=lambda s: (s.quality_rank, -s.baseline_seconds))
        fits[best.key].recommended = True
    return fits


def derive_tier(profile: HardwareProfile, specs) -> str:
    """The tier label, derived from fit results rather than a threshold table.

    Deriving it means the label can never contradict the catalog. The tier drives
    UI copy and grouping only — it never gates a model on its own.
    """
    if profile.device == "cpu":
        return "cpu"
    fits = {s.key for s in specs if fit(profile, s).fits}
    if any(s.arch == "flux" and s.key in fits for s in specs):
        return "ultra"
    sdxl = [s for s in specs if s.arch == "sdxl"]
    if sdxl and all(s.key in fits for s in sdxl):
        # Headroom for larger batches and resolutions, not just for the weights.
        if profile.budget_gb >= 20:
            return "high"
        return "mid"
    if any(s.key in fits for s in sdxl):
        return "mid"
    return "low"


# ------------------------------------------------------------- persistence ---

def save(profile: HardwareProfile, path=None) -> Path:
    path = Path(path or PROFILE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(profile), indent=2))
    return path


def load(path=None) -> HardwareProfile | None:
    """The stored profile, or None if there isn't a usable one.

    A missing file, unreadable JSON, a stale schema_version, or a field set that
    no longer matches the dataclass all return None — which the UI reads as
    "never scanned" and turns into the first-run wizard.
    """
    path = Path(path or PROFILE_PATH)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if data.get("schema_version") != SCHEMA_VERSION:
        return None
    try:
        return HardwareProfile(**data)
    except TypeError:
        return None
