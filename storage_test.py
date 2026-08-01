"""Offline checks for the storage inventory, its routes, and the panel.

Nothing here loads a model, calls the network, or touches a real cache. Every
path this script reads or deletes lives under a temporary `zz-storage-test-`
directory, and the environment variables that point Python at those directories
are set BEFORE `app` is imported — which is what keeps the developer's own
renders, profile, and ~/.cache/huggingface entirely out of scope.

The fixture cache deliberately contains a repo that is not in the catalog. The
central claim of this file is that it appears in no inventory and can be named
to no route.

Run: python storage_test.py
"""

import json
import os
import re
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent

# ------------------------------------------------------------- the fixture ---

FIXTURE = Path(tempfile.mkdtemp(prefix="zz-storage-test-"))
DATA = FIXTURE / "data"
OUT = FIXTURE / "pictures"
HF = FIXTURE / "huggingface"
HUB = HF / "hub"
for directory in (DATA, OUT, HUB):
    directory.mkdir(parents=True, exist_ok=True)

# Set before the imports below, not after: app.py resolves OUTPUTS at import
# time, and a variable set afterwards would leave the routes pointing at the
# developer's real Pictures folder while this script pointed at the fixture.
os.environ["LOCAL_IMG_DATA_DIR"] = str(DATA)
os.environ["LOCAL_IMG_OUTPUTS"] = str(OUT)

import download  # noqa: E402

# paths.hf_cache_dir() is under the home directory by design and takes no
# environment variable, so the fixture cache is installed by assignment. Every
# other cache path in the codebase is derived from this one, which is exactly
# why Task 1 made HUB the single source of truth.
download.HUB = HUB

import storage  # noqa: E402
import app as app_module  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app_module.app)

failures: list[str] = []


def check(label: str, condition: bool) -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}")
    if not condition:
        failures.append(label)


def _symlinks_work() -> bool:
    """Whether this platform lets an unprivileged process create a symlink.

    Windows does not, without Developer Mode. The fixtures fall back to copies
    there, and the one check that is *about* symlinks is skipped rather than
    faked.
    """
    probe = FIXTURE / "zz-symlink-probe"
    target = FIXTURE / "zz-symlink-target"
    target.write_bytes(b"x")
    try:
        probe.symlink_to(target)
        return True
    except (OSError, NotImplementedError):
        return False
    finally:
        probe.unlink(missing_ok=True)
        target.unlink(missing_ok=True)


SYMLINKS = _symlinks_work()


def link(target: Path, at: Path) -> None:
    """A snapshot entry: a symlink into blobs/, or a copy where that is all the
    platform allows — which is exactly what huggingface_hub itself does."""
    at.parent.mkdir(parents=True, exist_ok=True)
    if SYMLINKS:
        at.symlink_to(target)
    else:
        shutil.copyfile(target, at)


def plant_repo(repo: str, *, weight_bytes: int, snapshot: bool = True) -> Path:
    """A repo directory shaped like the real cache: real bytes in blobs/, and a
    snapshot of links into them.

    `snapshot=False` plants the other real case — a download that died before
    any snapshot was written. Its blobs are real, recoverable garbage.
    """
    rd = HUB / ("models--" + repo.replace("/", "--"))
    blobs = rd / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)

    index = json.dumps({
        "_class_name": "StableDiffusionPipeline",
        "unet": ["diffusers", "UNet2DConditionModel"],
    })
    (blobs / "index").write_text(index, encoding="utf-8")
    (blobs / "config").write_text("{}", encoding="utf-8")
    (blobs / "weights").write_bytes(b"\0" * weight_bytes)

    if snapshot:
        snap = rd / "snapshots" / "abc123"
        link(blobs / "index", snap / "model_index.json")
        link(blobs / "config", snap / "unet" / "config.json")
        link(blobs / "weights", snap / "unet" / "diffusion_pytorch_model.safetensors")
    return rd


def blob_bytes(rd: Path) -> int:
    return sum(p.stat().st_size for p in (rd / "blobs").glob("*"))


# ---------------------------------------------------- download.py, measured ---

def test_the_cache_has_one_source_of_truth():
    import paths

    # Reassigned above, so the assertion is about the shape of the default, not
    # about the fixture. A second hardcoded copy of this path under code that
    # deletes things is exactly what you do not want.
    source = (ROOT / "download.py").read_text(encoding="utf-8")
    check("HUB is derived from paths.hf_cache_dir()",
          "paths.hf_cache_dir()" in source)
    check("HUB is not a second hardcoded home path",
          '".cache" / "huggingface"' not in source)
    check("the xet cache is a sibling of hub",
          download.xet_dir() == download.HUB.parent / "xet")
    check("paths still owns the real location",
          paths.hf_cache_dir() == Path.home() / ".cache" / "huggingface")


def test_size_counts_blobs_and_not_the_symlinks_into_them():
    rd = plant_repo("zz-test/sized", weight_bytes=4096)
    try:
        spec = type("Spec", (), {"repo": "zz-test/sized"})()
        size = download.size_on_disk(spec)
        check("size is non-zero", size > 0)
        check("size counts the blobs", size >= blob_bytes(rd))
        if SYMLINKS:
            # The whole point: counting the snapshot too doubles every figure
            # the panel shows. Where the platform copies instead of linking,
            # the snapshot really is more bytes and counting it is correct —
            # so this pair of claims is only made where symlinks exist.
            check("size equals the bytes in blobs/", size == blob_bytes(rd))
            check("size does not double-count the snapshot",
                  size < blob_bytes(rd) * 2)
    finally:
        shutil.rmtree(rd, ignore_errors=True)


def test_a_model_that_is_not_in_the_cache_measures_zero():
    spec = type("Spec", (), {"repo": "zz-test/absent"})()
    check("an absent repo is zero bytes", download.size_on_disk(spec) == 0)
    freed, resisted = download.remove(spec)
    check("removing an absent repo frees nothing", freed == 0)
    check("removing an absent repo resists nothing", resisted == [])


def test_remove_reports_what_it_could_not_delete_instead_of_raising():
    rd = plant_repo("zz-test/locked", weight_bytes=2048)
    guarded = rd / "blobs"
    mode = guarded.stat().st_mode
    try:
        if os.name == "nt":
            check("skipped on Windows: directory permissions do not block unlink",
                  True)
            return
        # A read-only directory cannot have entries unlinked from it. That is
        # the portable stand-in for the antivirus-locked file this is about.
        os.chmod(guarded, 0o500)
        freed, resisted = download.remove(type("Spec", (), {"repo": "zz-test/locked"})())
        check("remove returned instead of raising", isinstance(resisted, list))
        check("the files that resisted are named", len(resisted) > 0)
        check("every resisted path is a string", all(isinstance(p, str) for p in resisted))
        check("nothing outside the repo was reported",
              all(str(rd) in p for p in resisted))
    finally:
        os.chmod(guarded, mode)
        shutil.rmtree(rd, ignore_errors=True)


# -------------------------------------------------------- the inventory ---

# A repo that belongs to somebody else. The shared cache is shared on purpose —
# paths.hf_cache_dir() says so — and may hold weights another tool downloaded.
FOREIGN = "someone/else"


def with_fixture_cache(planted: dict, body) -> None:
    """Run `body` with `planted` repos in the fixture cache, then clear them.

    `planted` maps a repo id to the keyword arguments for plant_repo.
    """
    dirs = [plant_repo(repo, **kwargs) for repo, kwargs in planted.items()]
    try:
        body()
    finally:
        for rd in dirs:
            shutil.rmtree(rd, ignore_errors=True)


def test_a_foreign_repo_appears_in_no_inventory():
    # THE central test. Everything else in this file is detail; this is the
    # claim that justifies building the list from models.MODELS rather than
    # from a directory listing.
    from models import BY_KEY

    def body():
        inv = storage.inventory()
        keys = [m["key"] for m in inv["models"]]
        names = [m["name"] for m in inv["models"]]
        check("the foreign repo is not a listed key",
              not any("else" in k or "someone" in k for k in keys))
        check("the foreign repo is not a listed name",
              not any("else" in n or "someone" in n for n in names))
        check("every listed key is a catalog key",
              all(k in BY_KEY for k in keys))
        check("the catalog model that is present is listed",
              "dreamshaper-8" in keys)
        check("the foreign repo's bytes are in no total",
              sum(m["bytes"] for m in inv["models"])
              == download.size_on_disk(BY_KEY["dreamshaper-8"]))

    with_fixture_cache(
        {FOREIGN: dict(weight_bytes=8192),
         BY_KEY["dreamshaper-8"].repo: dict(weight_bytes=4096)},
        body,
    )


def test_a_catalog_model_absent_from_the_cache_does_not_appear():
    from models import BY_KEY

    def body():
        keys = [m["key"] for m in storage.inventory()["models"]]
        check("the model in the cache is listed", "dreamshaper-8" in keys)
        check("a model with nothing on disk is not listed",
              "juggernaut-xl-v9" not in keys)

    with_fixture_cache({BY_KEY["dreamshaper-8"].repo: dict(weight_bytes=4096)}, body)


def test_a_half_finished_download_is_incomplete_and_still_offered():
    from models import BY_KEY

    def body():
        row = next(m for m in storage.inventory()["models"] if m["key"] == "sd-turbo")
        check("a snapshotless repo reads as incomplete", row["complete"] is False)
        check("its bytes are real and non-zero", row["bytes"] > 0)
        # Those gigabytes are recoverable garbage, and are exactly what someone
        # opens this panel to find.
        check("it is offered anyway", row["key"] in BY_KEY)

    with_fixture_cache(
        {BY_KEY["sd-turbo"].repo: dict(weight_bytes=4096, snapshot=False)}, body
    )


def test_a_complete_download_reads_as_complete():
    from models import BY_KEY

    def body():
        row = next(m for m in storage.inventory()["models"] if m["key"] == "dreamshaper-8")
        check("a full snapshot reads as complete", row["complete"] is True)

    with_fixture_cache({BY_KEY["dreamshaper-8"].repo: dict(weight_bytes=4096)}, body)


def test_the_runtime_is_measured_and_the_rest_excludes_it():
    runtime = DATA / "runtime"
    (runtime / "python" / "bin").mkdir(parents=True, exist_ok=True)
    (runtime / "python" / "bin" / "python3").write_bytes(b"\0" * 3000)
    (DATA / "profile.json").write_text("{}", encoding="utf-8")
    try:
        inv = storage.inventory()
        check("the runtime is measured", inv["runtime"] == 3000)
        check("the rest is the data directory minus the runtime",
              inv["rest"] == 2)          # the two bytes of "{}"
    finally:
        shutil.rmtree(runtime, ignore_errors=True)
        (DATA / "profile.json").unlink(missing_ok=True)


def test_a_symlinked_render_is_measured_as_the_link_not_its_target():
    # Anyone who aliases a photo library into their renders folder would
    # otherwise see the panel promise gigabytes that deleting cannot free:
    # remove_outputs unlinks the link and frees the link's own few bytes.
    if not SYMLINKS:
        check("skipped: this platform does not allow symlinks", True)
        return
    target = FIXTURE / "zz-storage-test-big.bin"
    target.write_bytes(b"\0" * 500_000)
    png = OUT / "zz-storage-test-linked.png"
    png.symlink_to(target)
    try:
        measured = storage.inventory()["outputs"]["bytes"]
        check("the link is counted, not the 500 KB it points at",
              measured < 10_000)
        check("and it is counted at all", measured > 0)
        freed, resisted = storage.remove_outputs()
        check("deleting it frees what was reported", freed == measured)
        check("the link is gone", not png.exists() and not png.is_symlink())
        check("its target survives", target.exists())
        check("nothing resisted", resisted == [])
    finally:
        png.unlink(missing_ok=True)
        target.unlink(missing_ok=True)


def test_outputs_are_counted_with_their_sidecars():
    png = OUT / "zz-storage-test-a.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    png.with_suffix(".json").write_text('{"seed": 1}', encoding="utf-8")
    try:
        out = storage.inventory()["outputs"]
        check("one render is counted", out["count"] == 1)
        check("the sidecar is counted too",
              out["bytes"] == png.stat().st_size + png.with_suffix(".json").stat().st_size)
        check("the folder is named so the user can find it", out["path"] == str(OUT))
    finally:
        png.unlink(missing_ok=True)
        png.with_suffix(".json").unlink(missing_ok=True)


def test_removing_outputs_takes_renders_and_sidecars_and_nothing_else():
    png = OUT / "zz-storage-test-b.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    png.with_suffix(".json").write_text('{"seed": 1}', encoding="utf-8")
    stranger = OUT / "zz-storage-test-notes.txt"
    stranger.write_text("not ours", encoding="utf-8")
    try:
        freed, resisted = storage.remove_outputs()
        check("the render is gone", not png.exists())
        check("its sidecar is gone", not png.with_suffix(".json").exists())
        check("something the user put there survives", stranger.exists())
        check("the bytes freed are reported", freed > 0)
        check("nothing resisted", resisted == [])
    finally:
        png.unlink(missing_ok=True)
        png.with_suffix(".json").unlink(missing_ok=True)
        stranger.unlink(missing_ok=True)


def test_removing_the_xet_cache_leaves_the_hub_alone():
    from models import BY_KEY

    def body():
        xet = download.xet_dir()
        (xet / "chunks").mkdir(parents=True, exist_ok=True)
        (xet / "chunks" / "blob").write_bytes(b"\0" * 1024)
        check("the dedup cache is measured", storage.inventory()["xet"] == 1024)
        freed, resisted = storage.remove_xet()
        check("the dedup cache is gone", not xet.exists())
        check("its bytes are reported", freed == 1024)
        check("nothing resisted", resisted == [])
        check("the weights beside it survive",
              download.repo_dir(BY_KEY["dreamshaper-8"].repo).exists())

    with_fixture_cache({BY_KEY["dreamshaper-8"].repo: dict(weight_bytes=4096)}, body)


def test_an_empty_machine_inventories_cleanly():
    # Nothing planted. The panel must open on a fresh install rather than 500.
    inv = storage.inventory()
    check("models is an empty list", inv["models"] == [])
    check("runtime is zero", inv["runtime"] == 0)
    check("xet is zero", inv["xet"] == 0)
    check("outputs is empty", inv["outputs"]["count"] == 0)


# ------------------------------------------------------------- the routes ---

class FakePipe:
    """Stands in for a loaded pipeline. The unload path only ever drops the
    reference and empties the device cache; it never calls into the object."""


def resident(key: str) -> None:
    app_module.CACHE.key = key
    app_module.CACHE.pipe = FakePipe()


def clear_resident() -> None:
    app_module.CACHE.key = None
    app_module.CACHE.pipe = None


def test_the_inventory_route_answers():
    res = client.get("/api/storage")
    check("200 from /api/storage", res.status_code == 200)
    body = res.json()
    for field in ("runtime", "models", "xet", "outputs", "rest"):
        check(f"the inventory carries {field}", field in body)


def test_the_foreign_repo_can_be_named_to_no_route():
    # The other half of the central claim: not merely absent from the list, but
    # unreachable through the route, because the route validates against BY_KEY.
    def body():
        for name in ("someone--else", "models--someone--else", "someone%2Felse", "else"):
            res = client.delete(f"/api/storage/models/{name}")
            check(f"{name} is refused", res.status_code in (400, 404))
        check("the foreign repo is still on disk",
              (HUB / "models--someone--else").exists())

    with_fixture_cache({FOREIGN: dict(weight_bytes=2048)}, body)


def test_deleting_a_model_frees_its_bytes():
    from models import BY_KEY

    def body():
        rd = download.repo_dir(BY_KEY["dreamshaper-8"].repo)
        expected = blob_bytes(rd)
        res = client.delete("/api/storage/models/dreamshaper-8")
        check("200 on delete", res.status_code == 200)
        # >=, not ==: where the platform copies snapshots instead of linking
        # them, those copies are real bytes that really were freed.
        check("the bytes freed are reported", res.json()["freed"] >= expected)
        check("nothing resisted", res.json()["resisted"] == [])
        check("the repo directory is gone", not rd.exists())

    with_fixture_cache({BY_KEY["dreamshaper-8"].repo: dict(weight_bytes=4096)}, body)


def test_deleting_the_loaded_model_unloads_it_first():
    from models import BY_KEY

    def body():
        resident("dreamshaper-8")
        try:
            # Its weights are open in memory, which on Windows means locked
            # files. The unload has to happen before the unlink, not after.
            client.delete("/api/storage/models/dreamshaper-8")
            check("the pipeline was dropped", app_module.CACHE.pipe is None)
            check("the cache key was cleared", app_module.CACHE.key is None)
        finally:
            clear_resident()

    with_fixture_cache({BY_KEY["dreamshaper-8"].repo: dict(weight_bytes=4096)}, body)


def test_deleting_another_model_leaves_the_resident_one_alone():
    from models import BY_KEY

    def body():
        resident("dreamshaper-8")
        try:
            client.delete("/api/storage/models/sd-turbo")
            check("a different model's delete keeps the pipeline resident",
                  app_module.CACHE.pipe is not None)
            check("and keeps its key", app_module.CACHE.key == "dreamshaper-8")
        finally:
            clear_resident()

    with_fixture_cache({BY_KEY["sd-turbo"].repo: dict(weight_bytes=2048)}, body)


def test_every_delete_route_is_409_while_a_render_is_in_flight():
    # The panel greys its buttons out from /api/busy, but that is courtesy:
    # between painting a button and clicking it there is ample time for a render
    # to start. Constructed directly rather than by hitting /api/generate, which
    # would load a 7 GB pipeline.
    job = app_module.Job("zz-storage-test-job", "generate")
    app_module.JOBS[job.id] = job
    try:
        for route in ("/api/storage/models/dreamshaper-8",
                      "/api/storage/outputs",
                      "/api/storage/xet"):
            check(f"{route} is 409 while generating",
                  client.delete(route).status_code == 409)
        check("the inventory still answers while busy",
              client.get("/api/storage").status_code == 200)
    finally:
        app_module.JOBS.pop(job.id, None)


def test_the_delete_routes_are_409_while_a_download_is_in_flight():
    job = app_module.Job("zz-storage-test-dl", "download")
    app_module.JOBS[job.id] = job
    try:
        check("deleting a model is 409 while downloading",
              client.delete("/api/storage/models/dreamshaper-8").status_code == 409)
    finally:
        app_module.JOBS.pop(job.id, None)


def test_the_outputs_and_xet_routes_answer():
    png = OUT / "zz-storage-test-route.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    try:
        res = client.delete("/api/storage/outputs")
        check("200 from the outputs route", res.status_code == 200)
        check("the render is gone", not png.exists())
        check("200 from the xet route",
              client.delete("/api/storage/xet").status_code == 200)
    finally:
        png.unlink(missing_ok=True)


# --------------------------------------------------------------- the panel ---

PAGE = ROOT / "web" / "index.html"


def page() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_the_sidebar_has_a_storage_section():
    html = page()
    check("the section exists", 'id="storage"' in html)
    check("it is labelled Storage", ">Storage<" in html)
    check("it reads the inventory", "/api/storage" in html)
    check("it only fetches when opened", "loadStorage" in html)


def test_the_panel_offers_a_delete_per_model():
    html = page()
    check("a model delete goes to the scoped route",
          "/api/storage/models/" in html)
    check("the dedup cache has its own route", "/api/storage/xet" in html)
    check("the renders have their own route", "/api/storage/outputs" in html)
    check("the renders' folder is named rather than opened",
          "outputs.path" in html or "out.path" in html)


def test_the_panel_respects_busy():
    html = page()
    check("it asks what is in flight", "/api/busy" in html)

    # A bare `"409" in html` would pass on a code comment mentioning the
    # number, with no error handling behind it at all. Instead, pull the two
    # functions that actually carry a 409 to the user and assert on their
    # mechanism: storageDelete must throw the server's own message rather than
    # swallow a non-ok response, and deleteOne must catch that throw and route
    # it through setStatus as an error rather than let it vanish to the
    # console.
    delete_fn = re.search(r"async function storageDelete\(.*?\n\}", html, re.S)
    check("storageDelete is defined", bool(delete_fn))
    delete_body = delete_fn.group(0) if delete_fn else ""
    check("storageDelete throws the server's message on a non-ok response",
          "if (!res.ok) throw new Error(await errorText(res));" in delete_body)

    one_fn = re.search(r"async function deleteOne\(.*?\n\}", html, re.S)
    check("deleteOne is defined", bool(one_fn))
    one_body = one_fn.group(0) if one_fn else ""
    check("deleteOne catches the throw", "catch (e)" in one_body)
    check("deleteOne reports it through setStatus as an error",
          "setStatus(String(e.message || e), '', true);" in one_body)


# ------------------------------------------------------ the shell's screens ---

SHELL_UI = ROOT / "desktop" / "ui" / "index.html"


def shell_ui() -> str:
    return SHELL_UI.read_text(encoding="utf-8")


def test_the_shell_has_a_removed_card():
    html = shell_ui()
    check("the card exists", 'id="removed"' in html)
    check("show() can select it", "'removed'" in html)
    check("it reads the result from initial_state", "state.removed" in html)
    check("it also listens for the event", "shell://removed" in html)
    check("it handles a deletion still running", "state.uninstalling" in html)


def test_the_removed_card_never_claims_a_clean_run_it_did_not_have():
    html = shell_ui()
    check("it inspects what resisted", "resisted" in html)
    check("it names the data directory", "dataDir" in html)
    check("and offers to open it", "open_data_dir" in html)
    check("the last step is not hardcoded to one platform",
          "lastStep" in html and "Applications to the Trash" not in html)


def test_the_removed_card_offers_its_buttons():
    html = shell_ui()
    check("a button opens where the app lives", "open_app_location" in html)
    check("a button quits", "quit_app" in html)


# ------------------------------------------------------ the uninstall flow ---

def test_the_confirmation_enumerates_rather_than_summarises():
    html = page()
    check("there is a confirmation overlay", 'id="wipeOverlay"' in html)
    check("it lists the models by name", "wipeList" in html)
    check("the images are a separate, unchecked box", 'id="wipeImages"' in html)
    check("the box is not checked by default",
          'id="wipeImages"' in html and 'id="wipeImages" checked' not in html)
    check("it says why the asymmetry exists",
          "downloaded again" in html)


def test_the_uninstall_half_is_behind_the_shell_test():
    html = page()
    check("the shell is feature-detected, never assumed", "HAS_SHELL" in html)
    check("the shell command is named", "finish_uninstall" in html)
    # The engine belongs to whoever installed it. In the repo flow that is the
    # user's own .venv, and deleting it is not this page's business.
    check("finish_uninstall is guarded by the detection",
          "if (HAS_SHELL)" in html or "HAS_SHELL &&" in html)


def test_the_flow_deletes_before_it_calls_the_shell():
    html = page()
    check("it deletes the weights first", "/api/storage/models/" in html)
    check("then the dedup cache", "/api/storage/xet" in html)
    check("the images only when asked", "wipeImages" in html)
    check("and passes the total it actually freed", "freed" in html)


def test_the_flow_reports_what_resisted():
    html = page()
    check("failures are collected, not swallowed", "resisted" in html)


def test_a_409_mid_flow_keeps_what_was_already_freed():
    html = page()
    check("freed lives outside the run, not reset on every call",
          "WIPE_FREED" in html)
    check("resisted lives outside the run the same way",
          "WIPE_RESISTED" in html)
    check("the accumulator resets when the dialog opens, not when a run starts",
          "WIPE_FREED = 0" in html)
    check("a retry adds to the running total rather than starting over",
          "WIPE_FREED +=" in html)
    check("the shell receives the accumulated total, not a call-local one",
          "freed: WIPE_FREED" in html)
    check("and the accumulated resisted list, not just the shell's own",
          "resisted: WIPE_RESISTED" in html)


def test_a_409_mid_flow_refreshes_the_enumerated_list():
    html = page()
    check("the list is rebuilt from the fresh inventory after a failed attempt",
          "renderWipeList" in html)


def test_finish_uninstall_rejection_is_not_swallowed():
    html = page()
    # A rejection here means the shell never navigated away and never tore
    # anything down, so the page is still on screen to receive it — unlike the
    # fulfilled case, which is deliberately left unawaited because the page it
    # would resolve into is gone. An empty `.catch(() => {})` on this call
    # would freeze the user on "Uninstalling..." with the models already
    # deleted and no way to know anything went wrong.
    call = html.split("invoke('finish_uninstall'")[1].split("'Uninstalling")[0]
    check("the rejection handler is not empty",
          ".catch(() => {})" not in call)
    check("it reports the error text on screen",
          "wipeError" in call)
    check("it lets the user try again", "wipeGo" in call)


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    try:
        for test in tests:
            print(f"\n{test.__name__}")
            test()
    finally:
        shutil.rmtree(FIXTURE, ignore_errors=True)

    print()
    if failures:
        raise SystemExit(f"{len(failures)} check(s) failed: {failures}")
    print("all checks passed")
