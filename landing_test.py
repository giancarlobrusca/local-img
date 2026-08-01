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
    for cls in (".dl-hero", ".dl-others", ".dl-mobile", ".dl-sub"):
        check(f"style.css defines {cls}", cls in css)

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
        check(f"{page} exists", bool(html))
        links = release_links(html)
        check(f"{page} links to releases at all", bool(links))
        check(f"{page} sends every release link straight to a file",
              all(link.startswith(DOWNLOAD_BASE) for link in links))
        check(f"{page} offers exactly the four stable assets",
              {link[len(DOWNLOAD_BASE):] for link in links} == set(ASSETS))


def test_no_versioned_filenames() -> None:
    for name in ("index.html", "es/index.html", "download.js"):
        text = read(name)
        check(f"{name} exists", bool(text))
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
    check("download.js's hero links use the same prefix the HTML does",
          DOWNLOAD_BASE in js)
    for asset in ASSETS[:3]:      # the .deb is reached through the others line
        check(f"download.js knows {asset}", asset in js)


DATA_KEYS = ("verb", "others", "mobile", "mac", "windows", "linux")


def data_attr(html: str, key: str) -> str | None:
    match = re.search(rf'data-{key}="([^"]*)"', html)
    return match.group(1) if match else None


def test_hero_strings_are_translated() -> None:
    # The data-* attributes are the only script-facing copy: download.js
    # reads every visible hero string from them. A value that reverts to
    # English, or that goes missing, is invisible to every other check here.
    en = read("index.html")
    es = read("es/index.html")
    for key in DATA_KEYS:
        en_value = data_attr(en, key)
        es_value = data_attr(es, key)
        check(f"index.html declares data-{key}", en_value is not None)
        check(f"es/index.html declares data-{key}", es_value is not None)
        if en_value is not None and es_value is not None:
            check(f"data-{key} is translated, not copied from English",
                  en_value != es_value)


SITE = "https://local-img-seven.vercel.app/"


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
    pattern = re.compile(r"\d+(?:\.\d+)?\s*(?:GB|MB|s)\b")
    en = sorted(pattern.findall(read("index.html")))
    es = sorted(pattern.findall(read("es/index.html")))
    check("English and Spanish quote the same figures", en == es)
    if en != es:
        print(f"    only in English: {sorted(set(en) - set(es))}")
        print(f"    only in Spanish: {sorted(set(es) - set(en))}")


def _linearize(channel: float) -> float:
    """One sRGB channel, 0..1, to linear light. The WCAG 2.x formula."""
    if channel <= 0.03928:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def luminance(hex_colour: str) -> float:
    """Relative luminance of a #rrggbb string."""
    raw = hex_colour.lstrip("#")
    r, g, b = (int(raw[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return (0.2126 * _linearize(r)
            + 0.7152 * _linearize(g)
            + 0.0722 * _linearize(b))


def contrast(a: str, b: str) -> float:
    """The WCAG contrast ratio between two #rrggbb colours."""
    la, lb = luminance(a), luminance(b)
    lighter, darker = max(la, lb), min(la, lb)
    return (lighter + 0.05) / (darker + 0.05)


def theme_blocks(css: str) -> dict[str, dict[str, str]]:
    """The two explicit theme blocks, as {theme: {var: hex}}.

    Only the [data-theme] blocks are read. They are the ones that declare a
    complete palette, and unlike :root and the media query they are named, so
    a failure says which theme is wrong.
    """
    blocks: dict[str, dict[str, str]] = {}
    for theme in ("light", "dark"):
        match = re.search(
            r':root\[data-theme="' + theme + r'"\]\s*\{(.*?)\}', css, re.S)
        if not match:
            continue
        blocks[theme] = dict(
            re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})", match.group(1)))
    return blocks


# Every colour that ends up as text on the page background. 4.5:1 is WCAG AA
# for body-sized text, which is what --muted and --accent are used at.
ON_INK = ("text", "muted", "accent", "accent-2")


def test_the_palette_meets_contrast() -> None:
    blocks = theme_blocks(read("style.css"))
    check("style.css declares both [data-theme] palettes",
          set(blocks) == {"light", "dark"})
    for theme, variables in sorted(blocks.items()):
        ink = variables.get("ink")
        check(f"the {theme} palette declares --ink", ink is not None)
        if ink is None:
            continue
        for name in ON_INK:
            colour = variables.get(name)
            check(f"the {theme} palette declares --{name}", colour is not None)
            if colour is None:
                continue
            ratio = contrast(colour, ink)
            check(f"{theme} --{name} {colour} on --ink {ink} "
                  f"reaches 4.5:1 (it is {ratio:.2f}:1)", ratio >= 4.5)


def test_the_mark_is_inline_and_theme_aware() -> None:
    css = read("style.css")
    for theme, variables in sorted(theme_blocks(css).items()):
        for name in ("brand-raw", "brand-burnt"):
            check(f"the {theme} palette declares --{name}",
                  name in variables)

    # theme_blocks only reads the two [data-theme] blocks. The mark also has
    # to be coloured for the visitor who never touches the toggle, which means
    # :root and the prefers-color-scheme query too — four declarations each.
    # Without this, the mark renders with no fill on a first visit.
    for name in ("brand-raw", "brand-burnt"):
        check(f"--{name} is declared in all four theme blocks",
              css.count(f"--{name}:") == 4)

    for page in PAGES:
        html = read(page)
        # Inline, not <img src>: an external SVG cannot read the page's CSS
        # custom properties, so it could not follow the light/dark theme.
        check(f"{page} carries the mark inline",
              'viewBox="0 0 32 32"' in html)
        check(f"{page}'s mark takes its colour from the theme",
              "var(--brand-raw)" in html and "var(--brand-burnt)" in html)
        check(f"{page} links the favicon",
              'rel="icon"' in html and "favicon.svg" in html)


def test_the_pages_are_branded_ocre() -> None:
    for page in PAGES:
        html = read(page)
        title = re.search(r"<title>(.*?)</title>", html, re.S)
        check(f"{page} has a title", title is not None)
        if title is not None:
            check(f"{page}'s title names Ocre", "Ocre" in title.group(1))
        check(f"{page}'s og:title is Ocre",
              'property="og:title" content="Ocre"' in html)
        check(f"{page}'s eyebrow names Ocre", "<b>Ocre</b>" in html)


def test_the_macos_quote_is_still_literal() -> None:
    # macOS names the app in that dialog, and the app is still called
    # local-img. This test exists so that a careless find-and-replace of
    # local-img -> Ocre breaks the suite instead of the page: it would
    # promise the reader a dialog they will never see.
    for page in PAGES:
        html = read(page)
        quotes = re.findall(r'<p class="quote">(.*?)</p>', html, re.S)
        check(f"{page} still shows two security quotes", len(quotes) == 2)
        check(f"{page}'s macOS quote still says local-img",
              any("local-img" in quote for quote in quotes))


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
