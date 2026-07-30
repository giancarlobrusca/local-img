"""Offline checks for hardware detection, fit rules, and speed estimates.

Nothing here loads a model, touches the network, or needs a GPU. Importing app
pulls in torch, which takes a few seconds, but no generation endpoint is called.

Every fixture name carries the zz- prefix and is removed in a finally block, so
real renders in outputs/ and a real .local-img/profile.json are never at risk.

Run: python hardware_test.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent

# Every fixture name carries this prefix, and cleanup only touches files this
# script created — a real .local-img/profile.json is never at risk.
PREFIX = "zz-hardware-test-"
created: list[Path] = []

failures: list[str] = []


def check(label: str, condition: bool) -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}")
    if not condition:
        failures.append(label)


# ------------------------------------------------------------------ catalog ---

def test_catalog_columns():
    from models import BY_KEY, DEFAULT_MODEL, MODELS

    for spec in MODELS:
        check(f"{spec.key}: min_budget_gb > 0", spec.min_budget_gb > 0)
        check(f"{spec.key}: min_ram_gb > 0", spec.min_ram_gb > 0)
        check(f"{spec.key}: baseline_seconds > 0", spec.baseline_seconds > 0)
        check(f"{spec.key}: quality_rank > 0", spec.quality_rank > 0)
        check(f"{spec.key}: arch is known", spec.arch in ("sd15", "sdxl", "flux"))
        check(f"{spec.key}: no speed field", not hasattr(spec, "speed"))
        check(f"{spec.key}: no hardcoded recommendation", "recommended" not in spec.tags)

    keys = [m.key for m in MODELS]
    check("no duplicate keys", len(keys) == len(set(keys)))
    check("nine models", len(MODELS) == 9)
    ranks = [m.quality_rank for m in MODELS]
    check("quality ranks are unique", len(ranks) == len(set(ranks)))
    check("default model still exists", DEFAULT_MODEL in BY_KEY)

    for key in ("lcm-dreamshaper-7", "playground-v25", "flex-1-alpha", "shuttle-3-diffusion"):
        check(f"{key} is in the catalog", key in BY_KEY)

    # Distilled and EDM-trained models ship a scheduler that must not be replaced.
    for key in ("sdxl-turbo", "dreamshaper-xl-turbo", "lcm-dreamshaper-7",
                "playground-v25", "flex-1-alpha", "shuttle-3-diffusion"):
        check(f"{key} keeps its scheduler", BY_KEY[key].keep_scheduler)
    for key in ("dreamshaper-8", "juggernaut-xl-v9", "realvis-xl-v4"):
        check(f"{key} takes the DPM++ override", not BY_KEY[key].keep_scheduler)


# --------------------------------------------------------------------- fit ---

def _profile(device="mps", budget=11.8, ram=16.0, cores=16, perf=1.0, skipped=False):
    import hardware
    return hardware.HardwareProfile(
        schema_version=hardware.SCHEMA_VERSION,
        device=device,
        platform="darwin-arm64",
        chip="test chip",
        gpu_cores=cores,
        total_ram_gb=ram,
        budget_gb=budget,
        free_disk_gb=500.0,
        tier="mid",
        perf_factor=perf,
        partial=False,
        skipped=skipped,
        detected_at="2026-07-30T00:00:00",
    )


SD15 = {"lcm-dreamshaper-7", "dreamshaper-8"}
SDXL = {"sdxl-turbo", "dreamshaper-xl-turbo", "juggernaut-xl-v9",
        "realvis-xl-v4", "playground-v25"}
FLEX = {"flex-1-alpha"}
SHUTTLE = {"shuttle-3-diffusion"}

# name, profile kwargs, exactly-these-models-fit, tier, max_batch, recommended
MACHINES = [
    ("8 GB M1", dict(budget=4.9, ram=8.0, cores=8),
     SD15, "low", 1, "dreamshaper-8"),
    ("16 GB M1 Pro (reference)", dict(budget=11.8, ram=16.0, cores=16),
     SD15 | SDXL, "mid", 2, "dreamshaper-xl-turbo"),
    ("24 GB M4", dict(budget=18.0, ram=24.0, cores=10),
     SD15 | SDXL, "mid", 2, "dreamshaper-xl-turbo"),
    ("36 GB M4 Pro", dict(budget=27.0, ram=36.0, cores=20),
     SD15 | SDXL | FLEX, "ultra", 4, "flex-1-alpha"),
    ("64 GB M4 Max", dict(budget=48.0, ram=64.0, cores=40),
     SD15 | SDXL | FLEX | SHUTTLE, "ultra", 4, "shuttle-3-diffusion"),
    ("RTX 4090, 64 GB RAM", dict(device="cuda", budget=21.6, ram=64.0, cores=None, perf=6.0),
     SD15 | SDXL | FLEX, "ultra", 4, "flex-1-alpha"),
    ("RTX 3090, 32 GB RAM", dict(device="cuda", budget=21.6, ram=32.0, cores=None, perf=4.0),
     SD15 | SDXL, "high", 4, "dreamshaper-xl-turbo"),
    ("CPU, 16 GB RAM", dict(device="cpu", budget=9.6, ram=16.0, cores=None, perf=0.04),
     SD15, "cpu", 1, "dreamshaper-8"),
]


def test_fit_table():
    import hardware
    from models import MODELS

    for name, kwargs, expected, tier, batch, recommended in MACHINES:
        prof = _profile(**kwargs)
        fits = hardware.fit_all(prof, MODELS)
        actual = {k for k, f in fits.items() if f.fits}
        check(f"{name}: fitting set", actual == expected)
        check(f"{name}: tier {tier}", hardware.derive_tier(prof, MODELS) == tier)
        check(f"{name}: max_batch {batch}",
              all(f.max_batch == batch for f in fits.values()))
        picked = [k for k, f in fits.items() if f.recommended]
        check(f"{name}: recommends {recommended}", picked == [recommended])


def test_fit_reasons_name_both_numbers():
    import hardware
    from models import BY_KEY

    prof = _profile(budget=11.8, ram=16.0)
    shuttle = hardware.fit(prof, BY_KEY["shuttle-3-diffusion"])
    check("budget shortfall is reported", not shuttle.fits)
    check("budget reason has the requirement", "26" in shuttle.reason)
    check("budget reason has the machine", "11.8" in shuttle.reason)

    # Budget is satisfied here, so system RAM is the binding constraint.
    roomy = _profile(budget=27.0, ram=16.0)
    flex = hardware.fit(roomy, BY_KEY["flex-1-alpha"])
    check("ram shortfall is reported", not flex.fits)
    check("ram reason mentions system RAM", "system RAM" in flex.reason)
    check("ram reason has the requirement", "36" in flex.reason)
    check("ram reason has the machine", "16" in flex.reason)

    ok = hardware.fit(prof, BY_KEY["dreamshaper-xl-turbo"])
    check("a fitting model has an empty reason", ok.fits and ok.reason == "")


def test_cpu_doubles_the_model_cost():
    import hardware
    from models import BY_KEY

    # 9.6 GiB budget clears sd15's 4.0 doubled to 8.0, but not sdxl's 9.5 -> 19.0.
    prof = _profile(device="cpu", budget=9.6, ram=16.0, cores=None, perf=0.04)
    check("sd15 fits on cpu", hardware.fit(prof, BY_KEY["dreamshaper-8"]).fits)
    check("sdxl does not fit on cpu", not hardware.fit(prof, BY_KEY["sdxl-turbo"]).fits)
    check("cpu reason quotes the doubled cost",
          "19" in hardware.fit(prof, BY_KEY["sdxl-turbo"]).reason)


def test_no_profile_and_skipped_profile():
    import hardware
    from models import MODELS

    for label, prof in (("no profile", None), ("skipped", _profile(skipped=True))):
        fits = hardware.fit_all(prof, MODELS)
        check(f"{label}: every model fits", all(f.fits for f in fits.values()))
        check(f"{label}: no reasons", all(f.reason == "" for f in fits.values()))
        check(f"{label}: nothing recommended",
              not any(f.recommended for f in fits.values()))
        check(f"{label}: max_batch is 4", all(f.max_batch == 4 for f in fits.values()))


def test_max_batch_thresholds():
    import hardware
    check("batch 1 below 10 GiB", hardware.max_batch_for(9.9) == 1)
    check("batch 2 at 10 GiB", hardware.max_batch_for(10.0) == 2)
    check("batch 2 below 20 GiB", hardware.max_batch_for(19.9) == 2)
    check("batch 4 at 20 GiB", hardware.max_batch_for(20.0) == 4)


# ------------------------------------------------------------- persistence ---

def test_profile_round_trip():
    import hardware

    tmp = ROOT / f"{PREFIX}profile.json"
    created.append(tmp)
    prof = _profile()
    hardware.save(prof, tmp)
    check("save wrote the file", tmp.exists())
    check("load round-trips", hardware.load(tmp) == prof)


def test_bumped_schema_version_forces_a_rescan():
    import hardware

    tmp = ROOT / f"{PREFIX}stale.json"
    created.append(tmp)
    prof = _profile()
    hardware.save(prof, tmp)
    data = json.loads(tmp.read_text())
    data["schema_version"] = hardware.SCHEMA_VERSION + 1
    tmp.write_text(json.dumps(data))
    check("a stale schema_version loads as None", hardware.load(tmp) is None)


def test_missing_and_corrupt_profiles_load_as_none():
    import hardware

    tmp = ROOT / f"{PREFIX}corrupt.json"
    created.append(tmp)
    check("a missing profile loads as None", hardware.load(ROOT / f"{PREFIX}absent.json") is None)
    tmp.write_text("{not json")
    check("a corrupt profile loads as None", hardware.load(tmp) is None)


def test_profile_dir_is_gitignored():
    ignore = (ROOT / ".gitignore").read_text()
    check(".local-img/ is gitignored", ".local-img/" in ignore)


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    try:
        for test in tests:
            print(f"\n{test.__name__}")
            test()
    finally:
        for path in created:
            path.unlink(missing_ok=True)

    print()
    if failures:
        raise SystemExit(f"{len(failures)} check(s) failed: {failures}")
    print("all checks passed")
