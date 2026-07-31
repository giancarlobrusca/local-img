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


DOWNLOAD_BASE = ("https://github.com/giancarlobrusca/local-img"
                 "/releases/latest/download/")

ASSETS = (
    "local-img-macos-arm64.dmg",
    "local-img-windows-x64.msi",
    "local-img-linux-x64.AppImage",
    "local-img-linux-x64.deb",
)

PAGES = ("index.html", "es/index.html")


def release_links(html: str) -> list[str]:
    return re.findall(r'href="([^"]*/local-img/releases/[^"]*)"', html)


def test_downloads_are_direct() -> None:
    for page in PAGES:
        html = read(page)
        if not html:
            continue          # es/index.html arrives in Task 5
        links = release_links(html)
        check(f"{page} links to releases at all", bool(links))
        check(f"{page} sends every release link straight to a file",
              all(link.startswith(DOWNLOAD_BASE) for link in links))
        check(f"{page} offers exactly the four stable assets",
              {link[len(DOWNLOAD_BASE):] for link in links} == set(ASSETS))


def test_no_versioned_filenames() -> None:
    for name in ("index.html", "es/index.html", "download.js"):
        text = read(name)
        if not text:
            continue          # both arrive in later tasks
        check(f"{name} names no versioned bundle", "local-img_" not in text)


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
