"""Offline checks for the landing pages in docs/.

Standard library only, so this runs with no install and no torch. It reads
the files as text; it does not start a browser or touch the network.

Run: python landing_test.py
"""

import re
from pathlib import Path

DOCS = Path(__file__).parent / "docs"

failures: list[str] = []


def check(label: str, condition: bool) -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}")
    if not condition:
        failures.append(label)


def read(name: str) -> str:
    path = DOCS / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


def test_stylesheet_is_shared() -> None:
    css = read("style.css")
    check("style.css exists and is not a stub", len(css) > 2000)
    check("style.css is CSS, not a page with tags in it", "</" not in css)

    en = read("index.html")
    check("the English page has no inline <style>", "<style" not in en)
    check("the English page links style.css",
          'href="style.css"' in en)


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for test in tests:
        print(f"\n{test.__name__}")
        test()

    print()
    if failures:
        raise SystemExit(f"{len(failures)} check(s) failed: {failures}")
    print("all checks passed")
