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


def test_detection_script_is_wired() -> None:
    en = read("index.html")
    check("the English page loads download.js",
          'src="download.js"' in en)
    check("the fallback grid keeps its id", 'id="downloads"' in en)

    js = read("download.js")
    check("download.js exists", len(js) > 200)
    check("download.js exposes the pure function for testing",
          "window.localImgPickPlatform" in js)
    check("download.js refuses to guess on touch devices",
          "maxTouchPoints" in js)
    for asset in ASSETS[:3]:      # the .deb is reached through the others line
        check(f"download.js knows {asset}", asset in js)


SITE = "https://giancarlobrusca.github.io/local-img/"


def test_each_page_declares_the_other() -> None:
    for page, lang, other in (("index.html", "en", "es/"),
                              ("es/index.html", "es", "../")):
        html = read(page)
        check(f"{page} exists", bool(html))
        check(f"{page} declares lang={lang}", f'<html lang="{lang}">' in html)
        check(f"{page} points at the English URL",
              f'hreflang="en" href="{SITE}"' in html)
        check(f"{page} points at the Spanish URL",
              f'hreflang="es" href="{SITE}es/"' in html)
        check(f"{page} names an x-default",
              f'hreflang="x-default" href="{SITE}"' in html)
        check(f"{page} links to its counterpart", f'href="{other}"' in html)

    es = read("es/index.html")
    check("the Spanish page links the shared stylesheet",
          'href="../style.css"' in es)
    check("the Spanish page loads the shared script",
          'src="../download.js"' in es)
    check("the Spanish page has no inline <style>", "<style" not in es)


def test_both_languages_state_the_same_numbers() -> None:
    # Model sizes, memory floors and timings are facts, not copy. A number
    # that drifts in one language is a page quietly lying to half its readers.
    pattern = re.compile(r"\d+(?:\.\d+)?\s*(?:GB|s)\b")
    en = sorted(pattern.findall(read("index.html")))
    es = sorted(pattern.findall(read("es/index.html")))
    check("English and Spanish quote the same figures", en == es)
    if en != es:
        print(f"    only in English: {sorted(set(en) - set(es))}")
        print(f"    only in Spanish: {sorted(set(es) - set(en))}")


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
