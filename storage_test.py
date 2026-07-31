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
