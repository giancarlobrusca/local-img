"""Offline checks for the two seams app.py grows for a parent process.

Importing app pulls in torch, which takes a few seconds, but nothing here
starts a server, loads a model, or touches the network.

Run: python shell_test.py
"""

import os
import subprocess
import sys

import app as app_module

failures: list[str] = []


def check(label: str, condition: bool) -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}")
    if not condition:
        failures.append(label)


def clear_env() -> None:
    for name in ("LOCAL_IMG_PORT", "LOCAL_IMG_PARENT_PID"):
        os.environ.pop(name, None)


# -------------------------------------------------------------- watchdog ---

def test_this_process_is_alive():
    check("our own pid is alive", app_module.pid_alive(os.getpid()) is True)
    check("our parent is alive", app_module.pid_alive(os.getppid()) is True)


def test_nonsense_pids_are_dead():
    check("pid 0 is dead", app_module.pid_alive(0) is False)
    check("a negative pid is dead", app_module.pid_alive(-1) is False)
    # -1 means "every process I may signal" to os.kill; reading it as alive
    # would keep a child running forever after its shell disappeared.


def test_a_reaped_child_reads_as_dead():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    check("a finished child is dead", app_module.pid_alive(child.pid) is False)


# ------------------------------------------------------------------ port ---

def test_port_defaults_to_7788():
    clear_env()
    check("no variable means 7788", app_module.chosen_port() == 7788)


def test_port_follows_the_variable():
    clear_env()
    os.environ["LOCAL_IMG_PORT"] = "51234"
    check("a valid port is used", app_module.chosen_port() == 51234)
    os.environ["LOCAL_IMG_PORT"] = "  51235  "
    check("whitespace is stripped", app_module.chosen_port() == 51235)


def test_an_unusable_port_falls_back_rather_than_crashing():
    # A shell that passed something broken should still get a working server;
    # refusing to start would leave the user with a blank window instead.
    clear_env()
    for bad in ("", "   ", "abc", "0", "-1", "65536", "99999999"):
        os.environ["LOCAL_IMG_PORT"] = bad
        check(f"{bad!r} falls back to 7788", app_module.chosen_port() == 7788)


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
