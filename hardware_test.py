"""Offline checks for hardware detection, fit rules, and speed estimates.

Nothing here loads a model, touches the network, or needs a GPU. Importing app
pulls in torch, which takes a few seconds, but no generation endpoint is called.

Every fixture name carries the zz- prefix and is removed in a finally block, so
real renders in outputs/ and a real .local-img/profile.json are never at risk.

Run: python hardware_test.py
"""

import json
import time
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


# ------------------------------------------------------------------ detect ---

def test_detect_on_this_machine():
    import hardware
    import app as app_module

    prof = hardware.detect(app_module.DEVICE)
    check("schema_version is stamped", prof.schema_version == hardware.SCHEMA_VERSION)
    check("device matches app", prof.device == app_module.DEVICE)
    check("budget_gb is positive", prof.budget_gb > 0)
    check("total_ram_gb is positive", prof.total_ram_gb > 0)
    check("free_disk_gb is positive", prof.free_disk_gb > 0)
    check("perf_factor is positive", prof.perf_factor > 0)
    check("tier is known", prof.tier in ("cpu", "low", "mid", "high", "ultra"))
    check("not skipped", prof.skipped is False)
    check("detected_at looks like ISO 8601", prof.detected_at[:4].isdigit())
    check("chip is non-empty", bool(prof.chip))
    print(f"       this machine: {prof.chip} · {prof.gpu_cores} cores · "
          f"{prof.total_ram_gb:.0f} GB · {prof.budget_gb:.1f} GB usable · {prof.tier}")


def test_detect_survives_broken_system_calls():
    import hardware

    def boom(*_args, **_kwargs):
        raise OSError("no such binary")


    real_run = hardware.subprocess.run
    try:
        hardware.subprocess.run = boom
        prof = hardware.detect("cpu")
    finally:
        hardware.subprocess.run = real_run

    check("detection did not propagate the failure", isinstance(prof, hardware.HardwareProfile))
    check("the profile is flagged partial", prof.partial is True)
    check("budget_gb is still positive", prof.budget_gb > 0)
    check("tier is still known", prof.tier in ("cpu", "low", "mid", "high", "ultra"))
    check("subprocess.run was restored", hardware.subprocess.run is real_run)


def test_detect_falls_back_when_something_unexpected_raises():
    import hardware

    def boom(*_args, **_kwargs):
        raise RuntimeError("something broke")

    real_derive_tier = hardware.derive_tier
    try:
        hardware.derive_tier = boom
        prof = hardware.detect("mps")
    finally:
        hardware.derive_tier = real_derive_tier

    check("detection returned a profile", isinstance(prof, hardware.HardwareProfile))
    check("the profile is flagged partial", prof.partial is True)
    check("budget_gb is still positive", prof.budget_gb > 0)
    check("tier is one of the known values", prof.tier in ("cpu", "low", "mid", "high", "ultra"))
    check("derive_tier was restored", hardware.derive_tier is real_derive_tier)


def test_skipped_profile():
    import hardware
    from models import MODELS

    prof = hardware.skipped_profile("mps")
    check("skipped is recorded", prof.skipped is True)
    check("skipped is not partial", prof.partial is False)
    check("skipped carries the device", prof.device == "mps")
    check("skipped has no tier claim", prof.tier == "unknown")
    fits = hardware.fit_all(prof, MODELS)
    check("skipped: everything fits", all(f.fits for f in fits.values()))
    check("skipped: nothing recommended", not any(f.recommended for f in fits.values()))


def test_cuda_perf_factors():
    import hardware
    check("4090 class", hardware.cuda_perf_factor("NVIDIA GeForce RTX 4090") == 6.0)
    check("5090 class", hardware.cuda_perf_factor("NVIDIA GeForce RTX 5090") == 6.0)
    check("3090 class", hardware.cuda_perf_factor("NVIDIA GeForce RTX 3090 Ti") == 4.0)
    check("4080 class", hardware.cuda_perf_factor("NVIDIA GeForce RTX 4080") == 4.0)
    check("4060 class", hardware.cuda_perf_factor("NVIDIA GeForce RTX 4060") == 2.0)
    check("unrecognized", hardware.cuda_perf_factor("NVIDIA A100-SXM4") == 3.0)


# ---------------------------------------------------------------- estimate ---

def test_estimate_scales_with_perf_factor_and_steps():
    import hardware
    from models import BY_KEY

    spec = BY_KEY["dreamshaper-xl-turbo"]      # baseline 40 s at 7 steps
    reference = hardware.estimate(_profile(perf=1.0), spec)
    check("reference machine reproduces the baseline", reference.seconds == 40.0)
    check("reference machine label", reference.label == "~40 s (estimated)")
    check("reference machine source", reference.source == "scaled")

    faster = hardware.estimate(_profile(perf=2.5), spec)
    check("2.5x machine is 2.5x faster", faster.seconds == 16.0)
    check("2.5x machine label", faster.label == "~16 s (estimated)")

    doubled = hardware.estimate(_profile(perf=2.0), spec, steps=14)
    check("double the steps, double the time", doubled.seconds == 40.0)


def test_estimate_switches_to_history_at_three_samples():
    import hardware
    from models import BY_KEY

    spec = BY_KEY["dreamshaper-xl-turbo"]
    prof = _profile(perf=1.0)

    two = hardware.estimate(prof, spec, history=[10.0, 12.0])
    check("two samples are not enough", two.source == "scaled")
    check("two samples keep the estimated label", two.label == "~40 s (estimated)")

    three = hardware.estimate(prof, spec, history=[10.0, 12.0, 50.0])
    check("three samples switch to history", three.source == "history")
    check("history uses the median, not the mean", three.seconds == 12.0)
    check("history label", three.label == "~12 s (your average)")


def test_estimate_without_a_profile_uses_the_reference_machine():
    import hardware
    from models import BY_KEY

    spec = BY_KEY["dreamshaper-xl-turbo"]
    for label, prof in (("no profile", None), ("skipped", _profile(skipped=True))):
        est = hardware.estimate(prof, spec)
        check(f"{label}: unscaled baseline", est.seconds == 40.0)
        check(f"{label}: reference label", est.label == "~40 s (reference machine)")
        check(f"{label}: reference source", est.source == "reference")

    # A measurement of the user's own machine outranks the fallback, even when
    # they declined the scan — three real timings are better evidence than none.
    est = hardware.estimate(_profile(skipped=True), spec, history=[9.0, 11.0, 13.0])
    check("history outranks a skipped profile", est.source == "history")
    check("history label after skipping", est.label == "~11 s (your average)")


def test_read_history_groups_by_model():
    import hardware

    out = ROOT / f"{PREFIX}outputs"
    out.mkdir(exist_ok=True)
    samples = [
        ("a", {"model": "dreamshaper-8", "seconds": 18.0, "steps": 25}),
        ("b", {"model": "dreamshaper-8", "seconds": 20.0, "steps": 25}),
        ("c", {"model": "sdxl-turbo", "seconds": 5.5, "steps": 3}),
        ("d", {"model": "sdxl-turbo"}),            # no seconds — ignored
        ("e", {"seconds": 3.0}),                   # no model — ignored
    ]
    written = []
    for stem, meta in samples:
        path = out / f"{PREFIX}{stem}.json"
        path.write_text(json.dumps(meta))
        written.append(path)
    (out / f"{PREFIX}bad.json").write_text("{not json")
    written.append(out / f"{PREFIX}bad.json")

    try:
        hist = hardware.read_history(out)
        check("groups by model key", sorted(hist) == ["dreamshaper-8", "sdxl-turbo"])
        check("collects every timing", sorted(hist["dreamshaper-8"]) == [18.0, 20.0])
        check("skips records with no seconds", hist["sdxl-turbo"] == [5.5])
    finally:
        for path in written:
            path.unlink(missing_ok=True)
        out.rmdir()


def test_read_history_on_a_missing_directory():
    import hardware
    check("a missing outputs dir is an empty history",
          hardware.read_history(ROOT / f"{PREFIX}absent-dir") == {})


# ------------------------------------------------------------------ routes ---

def _client_with_temp_profile():
    """A TestClient whose profile path points at a zz- fixture.

    hardware.load() and save() read PROFILE_PATH at call time, so patching the
    module global after import is enough — the developer's real
    .local-img/profile.json is never read or written. Returns a `restore`
    callable so callers can put the module global back in their own `finally`,
    matching the monkeypatch/restore pattern used elsewhere in this file (see
    test_detect_survives_broken_system_calls).
    """
    from fastapi.testclient import TestClient
    import hardware
    import app as app_module

    tmp = ROOT / f"{PREFIX}route-profile.json"
    tmp.unlink(missing_ok=True)
    created.append(tmp)
    real_path = hardware.PROFILE_PATH
    hardware.PROFILE_PATH = tmp

    def restore():
        hardware.PROFILE_PATH = real_path

    return TestClient(app_module.app), tmp, restore


def test_temp_profile_helper_restores_the_global():
    import hardware

    real_path = hardware.PROFILE_PATH
    _client, tmp, restore = _client_with_temp_profile()
    try:
        check("the global is patched to the fixture while in use",
              hardware.PROFILE_PATH == tmp)
    finally:
        restore()
    check("PROFILE_PATH was restored", hardware.PROFILE_PATH == real_path)


def test_models_route_without_a_profile():
    import hardware
    import app as app_module
    from models import DEFAULT_MODEL

    client, _, restore = _client_with_temp_profile()
    try:
        data = client.get("/api/models").json()
        check("profile is null before any scan", data["profile"] is None)
        check("falls back to the hardcoded default", data["default"] == DEFAULT_MODEL)
        check("every model fits", all(m["fits"] for m in data["models"]))
        check("nothing is recommended", not any(m["recommended"] for m in data["models"]))
        check("every model carries an estimate",
              all("label" in m["estimate"] for m in data["models"]))

        # With no profile, a model falls back to "reference" unless this machine's
        # own render history (outputs/*.json) already clears HISTORY_MIN samples —
        # in which case history correctly outranks the fallback (see hardware.py
        # estimate()). Derive the expectation instead of assuming a clean outputs/,
        # so this holds both on a fresh checkout and on a developer's machine.
        history = hardware.read_history(app_module.OUTPUTS)
        expected_source = {
            m["key"]: ("history" if len(history.get(m["key"], [])) >= hardware.HISTORY_MIN
                        else "reference")
            for m in data["models"]
        }
        check("estimates use history once recorded, reference otherwise",
              all(m["estimate"]["source"] == expected_source[m["key"]] for m in data["models"]))
        check("every model reports readiness",
              all(isinstance(m["ready"], bool) for m in data["models"]))
    finally:
        restore()


def test_scan_route_persists_and_populates():
    client, tmp, restore = _client_with_temp_profile()
    try:
        scanned = client.post("/api/hardware/scan", json={}).json()
        check("scan returns a profile", scanned["profile"] is not None)
        check("scan is not skipped", scanned["profile"]["skipped"] is False)
        check("scan wrote the profile", tmp.exists())

        data = client.get("/api/models").json()
        check("the profile survives into /api/models", data["profile"] is not None)
        check("the tier is set", data["profile"]["tier"] in ("cpu", "low", "mid", "high", "ultra"))
        fitting = [m for m in data["models"] if m["fits"]]
        check("at least one model fits this machine", len(fitting) >= 1)
        check("exactly one model is recommended",
              sum(1 for m in data["models"] if m["recommended"]) == 1)
        check("the default is the recommendation",
              data["default"] == next(m["key"] for m in data["models"] if m["recommended"]))
        check("non-fitting models carry a reason",
              all(m["reason"] for m in data["models"] if not m["fits"]))
    finally:
        restore()


def test_scan_route_can_persist_a_skip():
    client, tmp, restore = _client_with_temp_profile()
    try:
        data = client.post("/api/hardware/scan", json={"skip": True}).json()
        check("skip returns a profile", data["profile"] is not None)
        check("skip is recorded", data["profile"]["skipped"] is True)
        check("skip was persisted", tmp.exists())
        check("skip: everything fits", all(m["fits"] for m in data["models"]))
        check("skip: nothing recommended", not any(m["recommended"] for m in data["models"]))
        check("skip: max_batch is 4", all(m["max_batch"] == 4 for m in data["models"]))
    finally:
        restore()


def test_is_cached_reports_a_bool_for_every_model():
    import download
    from models import MODELS
    check("is_cached never raises",
          all(isinstance(download.is_cached(s), bool) for s in MODELS))


def test_is_cached_is_false_for_a_half_finished_download():
    """A model_index.json plus a dangling weight symlink — the real shape of a
    download that dropped mid-transfer: hf_hub_download creates the snapshot
    symlink first and writes the blob it points at afterward, so an
    interrupted pull leaves exactly this: a symlink whose target does not
    exist. Naively checking "the snapshot directory exists" would misread
    that as cached; is_cached must not.

    HUB is monkeypatched to a zz- fixture directory for the duration of this
    test and restored in the finally block — this never touches the real
    ~/.cache/huggingface tree.
    """
    import shutil
    import types
    import download

    real_hub = download.HUB
    fake_hub = ROOT / f"{PREFIX}hf-cache"
    snap = fake_hub / "models--zz-fake--half-done" / "snapshots" / "onlysnap"
    unet = snap / "unet"
    try:
        download.HUB = fake_hub
        unet.mkdir(parents=True)
        (snap / "model_index.json").write_text(json.dumps({
            "_class_name": "StableDiffusionPipeline",
            "unet": ["diffusers", "UNet2DConditionModel"],
        }))
        # The symlink hf_hub_download would have created, pointing at a blob
        # that never finished downloading — so the target does not exist.
        missing_blob = fake_hub / "does-not-exist.bin"
        weight = unet / "diffusion_pytorch_model.safetensors"
        weight.symlink_to(missing_blob)

        needed = download.needed_files(snap)
        check("the dangling weight file is part of what the pipeline needs",
              weight in needed)

        spec = types.SimpleNamespace(repo="zz-fake/half-done")
        check("is_cached is False for a dangling weight symlink",
              download.is_cached(spec) is False)
    finally:
        download.HUB = real_hub
        shutil.rmtree(fake_hub, ignore_errors=True)


# ---------------------------------------------------------------- download ---

def test_disk_shortfall_quotes_both_numbers():
    import app as app_module
    from models import BY_KEY

    spec = BY_KEY["dreamshaper-xl-turbo"]     # 6.9 GB -> needs 7.9 GB with headroom
    check("plenty of room is fine", app_module.disk_shortfall(spec, 500.0) is None)
    check("exactly enough is fine", app_module.disk_shortfall(spec, 8.0) is None)

    message = app_module.disk_shortfall(spec, 3.0)
    check("a shortfall is reported", message is not None)
    check("the message names the free space", "3.0" in message)
    check("the message names the requirement", "7.9" in message)


def test_download_route_rejects_an_unknown_key():
    # _client_with_temp_profile returns (client, tmp, restore) — see
    # test_temp_profile_helper_restores_the_global above; restore in finally.
    client, _, restore = _client_with_temp_profile()
    try:
        res = client.post("/api/download/zz-no-such-model")
        check("400 for an unknown key", res.status_code == 400)
    finally:
        restore()


def test_download_route_starts_a_job():
    import app as app_module

    client, _, restore = _client_with_temp_profile()
    # Never actually transfer 6.9 GB in a test.
    real_fetch = app_module.download.fetch
    calls: list = []
    job_id = ""
    try:
        app_module.download.fetch = lambda spec, **kw: calls.append(spec.key)
        res = client.post("/api/download/dreamshaper-xl-turbo")
        check("200 for a known key", res.status_code == 200)
        job_id = res.json().get("job", "")
        check("a job id comes back", isinstance(job_id, str) and bool(job_id))

        job = app_module.JOBS.get(job_id)
        check("the job is registered", job is not None)
        for _ in range(100):                 # up to 5 s
            if job.done:
                break
            time.sleep(0.05)
        check("the job finished", job.done)
        check("fetch was called once", calls == ["dreamshaper-xl-turbo"])
    finally:
        app_module.download.fetch = real_fetch
        app_module.JOBS.pop(job_id, None)
        restore()


# ----------------------------------------------------------- pipeline plan ---

def test_pipeline_plan_per_arch():
    import torch
    import app as app_module
    from models import BY_KEY

    sd15 = app_module.pipeline_plan(BY_KEY["dreamshaper-8"])
    check("sd15 uses StableDiffusionPipeline", sd15["pipeline"] == "StableDiffusionPipeline")
    check("sd15 disables the safety checker", sd15["disable_safety"] is True)
    check("sd15 stays on the device", sd15["offload"] is False)
    check("sd15 takes the DPM++ override", sd15["override_scheduler"] is True)
    check("sd15 uses the app dtype", sd15["dtype"] is app_module.DTYPE)

    sdxl = app_module.pipeline_plan(BY_KEY["dreamshaper-xl-turbo"])
    check("sdxl uses StableDiffusionXLPipeline", sdxl["pipeline"] == "StableDiffusionXLPipeline")
    check("sdxl has no safety checker to disable", sdxl["disable_safety"] is False)
    check("a turbo model keeps its scheduler", sdxl["override_scheduler"] is False)
    check("sdxl stays on the device", sdxl["offload"] is False)

    edm = app_module.pipeline_plan(BY_KEY["playground-v25"])
    check("playground keeps its EDM scheduler", edm["override_scheduler"] is False)

    flux = app_module.pipeline_plan(BY_KEY["shuttle-3-diffusion"])
    check("flux uses FluxPipeline", flux["pipeline"] == "FluxPipeline")
    check("flux runs bf16", flux["dtype"] is torch.bfloat16)
    check("flux is cpu-offloaded", flux["offload"] is True)
    check("flux keeps its scheduler", flux["override_scheduler"] is False)
    check("flux has no safety checker to disable", flux["disable_safety"] is False)


def test_download_picks_a_pipeline_class_for_every_arch():
    import download
    from models import MODELS

    names = {s.key: download._pipeline_cls(s).__name__ for s in MODELS}
    check("sd15 -> StableDiffusionPipeline",
          names["dreamshaper-8"] == "StableDiffusionPipeline")
    check("sdxl -> StableDiffusionXLPipeline",
          names["dreamshaper-xl-turbo"] == "StableDiffusionXLPipeline")
    check("flux -> FluxPipeline", names["shuttle-3-diffusion"] == "FluxPipeline")
    check("flux -> FluxPipeline (flex)", names["flex-1-alpha"] == "FluxPipeline")


def test_memory_hint_only_fires_on_memory_failures():
    import app as app_module
    from models import BY_KEY

    spec = BY_KEY["shuttle-3-diffusion"]
    for message in (
        "MPS backend out of memory (MPS allocated: 11.20 GB)",
        "CUDA out of memory. Tried to allocate 2.00 GiB",
        "Cannot allocate memory",
    ):
        hint = app_module.memory_hint(RuntimeError(message), spec)
        check(f"a memory failure gets a suggestion: {message[:24]}", "smaller model" in hint)
        check(f"the suggestion names the model: {message[:24]}", spec.name in hint)

    check("an ordinary failure gets no suggestion",
          app_module.memory_hint(ValueError("prompt is empty"), spec) == "")
    check("a missing spec is handled",
          app_module.memory_hint(RuntimeError("out of memory"), None) == "")


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
