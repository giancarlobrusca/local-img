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
import os
import platform as platform_mod
import shutil
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import paths

# Bumping this invalidates every stored profile and forces a re-scan.
SCHEMA_VERSION = 1


def profile_path() -> Path:
    """Re-exported so callers keep asking hardware.py where its own file is."""
    return paths.profile_path()

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

    if spec.arch == "flux" and profile.device == "cpu":
        # flux is loaded with enable_model_cpu_offload(), which offloads onto an
        # accelerator and raises when there isn't one. No amount of RAM fixes it,
        # so this is checked before the memory numbers.
        return Fit(
            fits=False,
            reason="needs a GPU; flux models are loaded with CPU offload, "
                   "which has no accelerator to offload onto on this machine",
            max_batch=batch,
            recommended=False,
        )
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


def least_demanding(specs):
    """The catalog entry closest to usable when nothing fits at all.

    Lowest GPU-visible requirement first, then lowest system RAM, then smallest
    download. Callers use it as the fallback pick on a machine where fit_all()
    recommends nothing and nothing fits — offering a model the same payload
    marks `fits: false` is unavoidable there, so offer the cheapest one.
    """
    return min(specs, key=lambda s: (s.min_budget_gb, s.min_ram_gb, s.download_gb))


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
    path = Path(path or profile_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(profile), indent=2), encoding="utf-8")
    return path


def load(path=None) -> HardwareProfile | None:
    """The stored profile, or None if there isn't a usable one.

    A missing file, unreadable JSON, a stale schema_version, or a field set that
    no longer matches the dataclass all return None — which the UI reads as
    "never scanned" and turns into the first-run wizard.
    """
    path = Path(path or profile_path())
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if data.get("schema_version") != SCHEMA_VERSION:
        return None
    try:
        return HardwareProfile(**data)
    except TypeError:
        return None


# ------------------------------------------------------------------ detect ---

def _gib(num_bytes: float) -> float:
    return num_bytes / (2 ** 30)


def _gb(num_bytes: float) -> float:
    return num_bytes / 1e9


def _try(fn, default, flags: list):
    """Run a probe. On any failure return `default` and record it as partial."""
    try:
        value = fn()
    except Exception:
        flags.append("x")
        return default
    if value is None:
        flags.append("x")
        return default
    return value


def _sysctl(name: str) -> str:
    out = subprocess.run(
        ["sysctl", "-n", name], capture_output=True, text=True, timeout=5, check=True
    )
    return out.stdout.strip()


def _gpu_cores() -> int | None:
    """Apple Silicon GPU core count, via system_profiler.

    Invoked with a 5 s timeout. If it is missing, slow, or returns unexpected
    JSON, the caller records `partial` and perf_factor falls back to the device
    default — the app never fails to start because of detection.
    """
    out = subprocess.run(
        ["system_profiler", "SPDisplaysDataType", "-json"],
        capture_output=True, text=True, timeout=5, check=True,
    )
    cards = json.loads(out.stdout).get("SPDisplaysDataType") or []
    for card in cards:
        cores = card.get("sppci_cores")
        if cores:
            return int(cores)
    return None


def _total_ram_gib(system: str) -> float:
    if system == "Darwin":
        return _gib(int(_sysctl("hw.memsize")))
    if system == "Windows":
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return _gib(status.ullTotalPhys)
    return _gib(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))


def _chip(system: str, device: str) -> str:
    if system == "Darwin":
        return _sysctl("machdep.cpu.brand_string")
    if device == "cuda":
        import torch

        return torch.cuda.get_device_name(0)
    return platform_mod.processor() or platform_mod.machine()


def _budget_gib(device: str, total_ram_gb: float) -> float:
    """Memory actually available for weights plus activations.

    Measured, not guessed. Metal's own answer on MPS; 90% of the card on CUDA;
    a fraction of system RAM on CPU, where the model also costs twice as much
    because app.py selects fp32 there.
    """
    import torch

    if device == "mps":
        return _gib(torch.mps.recommended_max_memory())
    if device == "cuda":
        return _gib(torch.cuda.get_device_properties(0).total_memory) * 0.9
    return total_ram_gb * CPU_BUDGET_FRACTION


def cuda_perf_factor(name: str) -> float:
    """Coarse by construction — this project has no NVIDIA hardware to calibrate
    against. These produce an order-of-magnitude first-run number that the UI
    always labels an estimate; recorded history replaces it after three renders.
    """
    lowered = name.lower()
    if any(tag in lowered for tag in ("4090", "5090")):
        return 6.0
    if any(tag in lowered for tag in ("4080", "5080", "3090")):
        return 4.0
    if any(tag in lowered for tag in ("3060", "4060", "5060", "3070", "4070")):
        return 2.0
    return 3.0


def _perf_factor(device: str, gpu_cores: int | None, chip: str) -> float:
    if device == "cpu":
        return CPU_PERF_FACTOR
    if device == "cuda":
        return cuda_perf_factor(chip)
    if gpu_cores:
        return gpu_cores / REFERENCE_GPU_CORES
    return 1.0


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def detect(device: str, specs=None) -> HardwareProfile:
    """Measure the machine. Never raises — a total failure returns a
    conservative profile with partial=True so the app always starts.

    `device` comes from app._pick_device(); passing it in rather than computing
    it here is what keeps this module free of an import cycle.
    """
    try:
        if specs is None:
            from models import MODELS as specs

        flags: list = []
        system = platform_mod.system()

        chip = _try(lambda: _chip(system, device), "unknown", flags)
        gpu_cores = None
        if system == "Darwin" and device == "mps":
            gpu_cores = _try(_gpu_cores, None, flags)
        total_ram_gb = _try(lambda: _total_ram_gib(system), 8.0, flags)
        budget_gb = _try(
            lambda: _budget_gib(device, total_ram_gb),
            total_ram_gb * CPU_BUDGET_FRACTION,
            flags,
        )
        # Weights land in the Hugging Face cache under home, which installed is
        # a different volume from the app bundle. Measure the volume that has to
        # hold them, not the one holding the code.
        free_disk_gb = _try(
            lambda: _gb(shutil.disk_usage(paths.disk_probe_dir()).free), 0.0, flags
        )

        profile = HardwareProfile(
            schema_version=SCHEMA_VERSION,
            device=device,
            platform=f"{system.lower()}-{platform_mod.machine()}",
            chip=chip,
            gpu_cores=gpu_cores,
            total_ram_gb=round(total_ram_gb, 2),
            budget_gb=round(budget_gb, 2),
            free_disk_gb=round(free_disk_gb, 1),
            tier="",                # filled in below, from the fit results
            perf_factor=round(_perf_factor(device, gpu_cores, chip), 3),
            partial=bool(flags),
            skipped=False,
            detected_at=_now(),
        )
        profile.tier = derive_tier(profile, specs)
        return profile
    except Exception:
        return _conservative(device)


def _conservative(device: str) -> HardwareProfile:
    """What detection returns when it fails outright. Fit rules still apply
    against these numbers — they are pessimistic, not absent.
    """
    total_ram_gb = 8.0
    try:
        total_ram_gb = _total_ram_gib(platform_mod.system())
    except Exception:
        pass
    return HardwareProfile(
        schema_version=SCHEMA_VERSION,
        device=device,
        platform=f"{platform_mod.system().lower()}-{platform_mod.machine()}",
        chip="unknown",
        gpu_cores=None,
        total_ram_gb=round(total_ram_gb, 2),
        budget_gb=round(total_ram_gb * CPU_BUDGET_FRACTION, 2),
        free_disk_gb=0.0,
        tier="cpu" if device == "cpu" else "low",
        perf_factor=CPU_PERF_FACTOR if device == "cpu" else 1.0,
        partial=True,
        skipped=False,
        detected_at=_now(),
    )


def skipped_profile(device: str) -> HardwareProfile:
    """A persisted record of a declined analysis, not a measurement.

    Every model reports fits=True with an empty reason, nothing is recommended,
    max_batch is 4, and estimates fall back to the reference machine. The banner
    offering the analysis stays until a real scan replaces this.
    """
    return HardwareProfile(
        schema_version=SCHEMA_VERSION,
        device=device,
        platform=f"{platform_mod.system().lower()}-{platform_mod.machine()}",
        chip="",
        gpu_cores=None,
        total_ram_gb=0.0,
        budget_gb=0.0,
        free_disk_gb=0.0,
        tier="unknown",
        perf_factor=1.0,
        partial=False,
        skipped=True,
        detected_at=_now(),
    )


# ---------------------------------------------------------------- estimate ---

# Below this many recorded renders the median is too noisy to trust.
HISTORY_MIN = 3


@dataclass
class Estimate:
    seconds: float
    label: str     # "~40 s (estimated)" | "~12 s (your average)" | "~40 s (reference machine)"
    source: str    # history | scaled | reference


def estimate(profile: HardwareProfile | None, spec, history=(), steps=None) -> Estimate:
    """Seconds per image, with the best evidence available.

    Three sources, strongest first:
      history   — the median of this machine's own recorded renders. Needs
                  HISTORY_MIN samples, and ignores `steps`: it reports what
                  actually happened, not a model of it.
      scaled    — baseline / perf_factor, adjusted for the step count.
      reference — the baseline itself, when nothing about this machine is known.
    """
    steps = steps or spec.steps
    history = list(history)

    if len(history) >= HISTORY_MIN:
        seconds = float(statistics.median(history))
        return Estimate(round(seconds, 1), f"~{seconds:.0f} s (your average)", "history")

    ratio = steps / spec.steps if spec.steps else 1.0
    if profile is None or profile.skipped:
        seconds = spec.baseline_seconds * ratio
        return Estimate(round(seconds, 1), f"~{seconds:.0f} s (reference machine)", "reference")

    factor = profile.perf_factor or 1.0
    seconds = spec.baseline_seconds / factor * ratio
    return Estimate(round(seconds, 1), f"~{seconds:.0f} s (estimated)", "scaled")


def read_history(outputs_dir) -> dict[str, list[float]]:
    """Recorded seconds per model, read from the render sidecars.

    Every generation already writes `model`, `steps` and `seconds` into a JSON
    sidecar next to its PNG. That makes the estimate self-correcting on any
    machine with no benchmark step and no extra bookkeeping.
    """
    outputs_dir = Path(outputs_dir)
    history: dict[str, list[float]] = {}
    if not outputs_dir.is_dir():
        return history
    for path in outputs_dir.glob("*.json"):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        key, seconds = meta.get("model"), meta.get("seconds")
        if isinstance(key, str) and isinstance(seconds, (int, float)):
            history.setdefault(key, []).append(float(seconds))
    return history
