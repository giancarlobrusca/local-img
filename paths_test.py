"""Offline checks for path resolution.

Nothing here imports app or torch. Every case sets environment variables and
puts them back, so running this script cannot change where the developer's own
profile or renders live.

Run: python paths_test.py
"""

import os
from pathlib import Path

import paths

ROOT = Path(__file__).parent

PREFIX = "zz-paths-test-"
failures: list[str] = []

VARS = ("LOCAL_IMG_DATA_DIR", "LOCAL_IMG_OUTPUTS")


def check(label: str, condition: bool) -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}")
    if not condition:
        failures.append(label)


def clear_env() -> None:
    for name in VARS:
        os.environ.pop(name, None)


def test_unset_matches_todays_repo_paths():
    clear_env()
    check("data dir is ./.local-img", paths.data_dir() == ROOT / ".local-img")
    check("profile is ./.local-img/profile.json",
          paths.profile_path() == ROOT / ".local-img" / "profile.json")
    check("outputs is ./outputs", paths.outputs_dir() == ROOT / "outputs")


def test_set_points_at_the_data_directory():
    clear_env()
    os.environ["LOCAL_IMG_DATA_DIR"] = f"/tmp/{PREFIX}data"
    os.environ["LOCAL_IMG_OUTPUTS"] = f"/tmp/{PREFIX}pics"
    check("data dir follows the variable",
          paths.data_dir() == Path(f"/tmp/{PREFIX}data"))
    check("profile sits in the data dir",
          paths.profile_path() == Path(f"/tmp/{PREFIX}data") / "profile.json")
    check("outputs follows its own variable",
          paths.outputs_dir() == Path(f"/tmp/{PREFIX}pics"))


def test_each_variable_resolves_independently():
    # A shell that set one and not the other must not drag the second one along.
    clear_env()
    os.environ["LOCAL_IMG_DATA_DIR"] = f"/tmp/{PREFIX}only-data"
    check("outputs stays repo-relative when only the data dir is set",
          paths.outputs_dir() == ROOT / "outputs")

    clear_env()
    os.environ["LOCAL_IMG_OUTPUTS"] = f"/tmp/{PREFIX}only-pics"
    check("data dir stays repo-relative when only outputs is set",
          paths.data_dir() == ROOT / ".local-img")


def test_blank_and_padded_values_are_treated_as_unset():
    clear_env()
    os.environ["LOCAL_IMG_DATA_DIR"] = "   "
    check("a blank variable falls back", paths.data_dir() == ROOT / ".local-img")
    os.environ["LOCAL_IMG_OUTPUTS"] = f"  /tmp/{PREFIX}padded  "
    check("surrounding whitespace is stripped",
          paths.outputs_dir() == Path(f"/tmp/{PREFIX}padded"))


def test_a_tilde_is_expanded():
    clear_env()
    os.environ["LOCAL_IMG_OUTPUTS"] = "~/Pictures/local-img"
    check("~ expands to the home directory",
          paths.outputs_dir() == Path.home() / "Pictures" / "local-img")


def test_the_hugging_face_cache_never_moves():
    clear_env()
    expected = Path.home() / ".cache" / "huggingface"
    check("cache is under home with no variables", paths.hf_cache_dir() == expected)
    os.environ["LOCAL_IMG_DATA_DIR"] = f"/tmp/{PREFIX}data"
    os.environ["LOCAL_IMG_OUTPUTS"] = f"/tmp/{PREFIX}pics"
    check("cache is under home in app mode too", paths.hf_cache_dir() == expected)


def test_the_disk_probe_is_the_volume_that_holds_the_weights():
    clear_env()
    probe = paths.disk_probe_dir()
    check("the probe directory exists", probe.is_dir())
    check("the probe is home, not the repo", probe == Path.home())


def test_hardware_reads_the_profile_through_paths():
    import hardware

    clear_env()
    check("the module constant is gone", not hasattr(hardware, "PROFILE_PATH"))
    check("hardware agrees with paths",
          hardware.profile_path() == paths.profile_path())
    os.environ["LOCAL_IMG_DATA_DIR"] = f"/tmp/{PREFIX}data"
    check("hardware follows the variable at call time",
          hardware.profile_path() == Path(f"/tmp/{PREFIX}data") / "profile.json")


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    try:
        for test in tests:
            print(f"\n{test.__name__}")
            test()
    finally:
        clear_env()

    print()
    if failures:
        raise SystemExit(f"{len(failures)} check(s) failed: {failures}")
    print("all checks passed")
