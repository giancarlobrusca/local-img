"""Offline checks for DELETE /api/outputs/{name}.

No model is loaded and no GPU work runs — importing app pulls in torch, which
takes a few seconds, but no generation endpoint is called.

Run: python delete_test.py
"""

from pathlib import Path

from fastapi import HTTPException
from fastapi.testclient import TestClient

import app as app_module

client = TestClient(app_module.app)
OUTPUTS = app_module.OUTPUTS
ROOT = Path(app_module.__file__).parent

# Every fixture name carries this prefix, and cleanup only touches files this
# script created — real renders in outputs/ are never at risk.
PREFIX = "zz-delete-test-"
created: list[Path] = []
failures: list[str] = []


def make(stem: str, sidecar: bool = True) -> Path:
    png = OUTPUTS / f"{PREFIX}{stem}.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    created.append(png)
    if sidecar:
        meta = png.with_suffix(".json")
        meta.write_text('{"seed": 1}')
        created.append(meta)
    return png


def check(label: str, condition: bool) -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}")
    if not condition:
        failures.append(label)


def test_deletes_png_and_sidecar():
    png = make("basic")
    res = client.delete(f"/api/outputs/{png.name}")
    check("200 on delete", res.status_code == 200)
    check("returns deleted name", res.json() == {"deleted": png.name})
    check("png is gone", not png.exists())
    check("sidecar is gone", not png.with_suffix(".json").exists())


def test_missing_name_is_404():
    res = client.delete(f"/api/outputs/{PREFIX}nope.png")
    check("404 for a name that does not exist", res.status_code == 404)


def test_traversal_is_404():
    # Percent-encoded so the client cannot normalize the path away before it
    # reaches the route. Either way the outcome is the same: 404, app.py alive.
    for encoded in ("%2e%2e%2fapp.py", "outputs%2f%2e%2e%2fapp.py"):
        res = client.delete(f"/api/outputs/{encoded}")
        check(f"404 for traversal {encoded}", res.status_code == 404)
    check("app.py still exists", (ROOT / "app.py").exists())


def test_guard_rejects_traversal_directly():
    # Over HTTP a traversal name decodes to a path with a slash, so the router
    # 404s it before the handler runs — that leaves the containment guard itself
    # unexercised. Call the handler directly to prove the guard, not the router.
    for bad in ("../app.py", "../outputs-old/x.png"):
        try:
            app_module.delete_output(bad)
            check(f"guard rejects {bad}", False)
        except HTTPException as exc:
            check(f"guard rejects {bad}", exc.status_code == 404)


def test_json_name_is_404():
    png = make("sidecar-target")
    meta = png.with_suffix(".json")
    res = client.delete(f"/api/outputs/{meta.name}")
    check("404 when aiming at a .json", res.status_code == 404)
    check("the .json survives", meta.exists())


def test_missing_sidecar_still_deletes():
    png = make("no-sidecar", sidecar=False)
    res = client.delete(f"/api/outputs/{png.name}")
    check("200 when the sidecar is absent", res.status_code == 200)
    check("png is gone", not png.exists())


if __name__ == "__main__":
    try:
        for test in (
            test_deletes_png_and_sidecar,
            test_missing_name_is_404,
            test_traversal_is_404,
            test_guard_rejects_traversal_directly,
            test_json_name_is_404,
            test_missing_sidecar_still_deletes,
        ):
            print(f"\n{test.__name__}")
            test()
    finally:
        for path in created:
            path.unlink(missing_ok=True)

    print()
    if failures:
        raise SystemExit(f"{len(failures)} check(s) failed: {failures}")
    print("all checks passed")
