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


# ------------------------------------------------------------- the gate ---

TOKEN = "zz-delete-test-token"


def gated_client():
    """A TestClient against a server that demands the session cookie.

    app.TOKEN is read by the middleware at call time, so setting the module
    global is enough. Returns a `restore` callable for the caller's `finally`.
    """
    previous = app_module.TOKEN
    app_module.TOKEN = TOKEN

    def restore():
        app_module.TOKEN = previous

    return TestClient(app_module.app), restore


def test_no_token_leaves_every_route_open():
    # This is the repo flow. With LOCAL_IMG_TOKEN unset nothing is gated, and
    # a foreign Origin is not rejected either — that is today's behaviour and
    # the whole point of making the gate opt-in.
    check("app.TOKEN is empty by default", app_module.TOKEN == "")
    res = client.get("/api/gallery", headers={"Origin": "http://evil.example"})
    check("the gallery answers with no cookie and a foreign Origin",
          res.status_code == 200)


def test_a_gated_server_refuses_a_cookieless_caller():
    gated, restore = gated_client()
    try:
        check("the gallery is 403 without the cookie",
              gated.get("/api/gallery").status_code == 403)
        check("delete is 403 without the cookie",
              gated.delete(f"/api/outputs/{PREFIX}whatever.png").status_code == 403)
        check("a wrong token in the query is 403",
              gated.get("/?token=wrong").status_code == 403)
    finally:
        restore()


def test_health_answers_before_the_cookie_exists():
    # The shell polls this to know the child is up, which happens before the
    # window has navigated anywhere and therefore before any cookie is set.
    gated, restore = gated_client()
    try:
        res = gated.get("/api/health")
        check("health is 200 with no cookie", res.status_code == 200)
        check("health reports ok", res.json().get("ok") is True)
    finally:
        restore()


def test_the_token_route_sets_the_cookie_and_redirects():
    gated, restore = gated_client()
    try:
        res = gated.get(f"/?token={TOKEN}&firstrun=1", follow_redirects=False)
        check("the token route redirects", res.status_code == 303)
        check("the redirect keeps firstrun", res.headers["location"] == "/?firstrun=1")
        cookie = res.headers["set-cookie"].lower()
        check("the cookie is HttpOnly", "httponly" in cookie)
        # Lax, not Strict, and the difference is the whole handoff. The shell
        # navigates the window from its own `tauri://localhost` page to
        # 127.0.0.1, which is a cross-site top-level navigation — and the 303
        # below is still part of it. WebKit withholds a Strict cookie from
        # every leg of such a navigation, so `/` arrives with no cookie, no
        # header and no query token, and the gate answers "missing session"
        # with no way out: each retry repeats the same handoff. Lax is defined
        # to ride exactly this request — a top-level GET — and nothing else.
        check("the cookie is SameSite=Lax", "samesite=lax" in cookie)
        check("the cookie carries the token", TOKEN in res.headers["set-cookie"])

        # httpx keeps the cookie in its jar, so the next call is a real
        # second request from the same browser.
        check("the redirect target answers once the cookie is set",
              gated.get("/").status_code == 200)
        check("the gallery answers once the cookie is set",
              gated.get("/api/gallery").status_code == 200)
    finally:
        restore()


def test_a_stale_cookie_does_not_shadow_a_fresh_handoff():
    """A cookie from an earlier launch must not be able to wedge the app.

    Cookies ignore the port, so every launch shares 127.0.0.1's jar with the
    one before it — while each launch mints a token of its own. Were the
    cookie consulted first, a leftover value would fail the gate on `/`, which
    is the one route that can replace it, and the shell's handoff would have
    no way back: every relaunch would answer "missing session" forever. So on
    `/` the query token wins, and everywhere else there is no query token to
    win with.
    """
    gated, restore = gated_client()
    # Sent as a header rather than seeded into httpx's jar: a jar entry with no
    # matching domain is silently dropped, and every check here would then pass
    # against a request that carried no stale cookie at all.
    stale = {"Cookie": f"{app_module.COOKIE_NAME}=token-from-an-earlier-launch"}
    try:
        check("a stale cookie authenticates nothing",
              gated.get("/api/gallery", headers=stale).status_code == 403)
        res = gated.get(f"/?token={TOKEN}", headers=stale, follow_redirects=False)
        check("the handoff redirects past a stale cookie", res.status_code == 303)
        check("the stale cookie is replaced",
              TOKEN in res.headers.get("set-cookie", ""))
        check("the page answers on the cookie that replaced it",
              gated.get("/").status_code == 200)
        # The precedence must not become a way in: a wrong token on / is still
        # a wrong token, cookie in the jar or not.
        check("a wrong query token is rejected even with a good cookie",
              gated.get("/?token=wrong").status_code == 403)
    finally:
        restore()


def test_a_foreign_origin_is_rejected_even_with_the_cookie():
    gated, restore = gated_client()
    try:
        gated.get(f"/?token={TOKEN}", follow_redirects=False)
        png = make("origin-guard")
        res = gated.delete(f"/api/outputs/{png.name}",
                           headers={"Origin": "http://127.0.0.1:9999"})
        check("a delete from another port is 403", res.status_code == 403)
        check("the render survives it", png.exists())
        check("a delete from a public page is 403",
              gated.delete(f"/api/outputs/{png.name}",
                           headers={"Origin": "https://evil.example"}).status_code == 403)
        check("the render still survives", png.exists())
        check("the page's own Origin is accepted",
              gated.delete(f"/api/outputs/{png.name}",
                           headers={"Origin": "http://testserver"}).status_code == 200)
    finally:
        restore()


def test_origin_allowed_is_a_predicate():
    # Called directly, because the interesting cases are the ones a browser
    # would never let a test send.
    allowed = app_module.origin_allowed
    check("no Origin is allowed", allowed(None, "http", "127.0.0.1:7788"))
    check("an empty Origin is allowed", allowed("", "http", "127.0.0.1:7788"))
    check("the same origin is allowed",
          allowed("http://127.0.0.1:7788", "http", "127.0.0.1:7788"))
    check("localhost against localhost is allowed",
          allowed("http://localhost:7788", "http", "localhost:7788"))
    check("another port is rejected",
          not allowed("http://127.0.0.1:9999", "http", "127.0.0.1:7788"))
    check("a public site is rejected",
          not allowed("https://evil.example", "http", "127.0.0.1:7788"))
    check("a sandboxed null Origin is rejected",
          not allowed("null", "http", "127.0.0.1:7788"))
    check("a file Origin is rejected",
          not allowed("file://", "http", "127.0.0.1:7788"))
    check("a missing Host with an Origin present is rejected",
          not allowed("http://127.0.0.1:7788", "http", None))
    check("a mismatched scheme is rejected",
          not allowed("https://127.0.0.1:7788", "http", "127.0.0.1:7788"))


def test_query_token_is_scoped_to_the_handoff_route():
    # / converts a query token into a cookie because a top-level navigation
    # cannot set a header. Nowhere else may accept it — a query string leaks
    # into Referer headers, browser history, shell history, and any future log.
    gated, restore = gated_client()
    try:
        check("a query token off / is 403",
              gated.get(f"/api/gallery?token={TOKEN}").status_code == 403)
        check("a query token on / still redirects",
              gated.get(f"/?token={TOKEN}", follow_redirects=False).status_code == 303)
    finally:
        restore()


def test_the_header_authenticates_without_a_cookie():
    # How the shell's own Rust code authenticates: a cross-origin page cannot
    # set a custom header without a CORS preflight, which the middleware
    # refuses, so this channel is open to the shell and shut to pages.
    gated, restore = gated_client()
    try:
        res = gated.get("/api/gallery", headers={"X-Local-Img-Token": TOKEN})
        check("the header authenticates", res.status_code == 200)
        res = gated.get("/api/gallery", headers={"X-Local-Img-Token": "wrong"})
        check("a wrong header value is 403", res.status_code == 403)
    finally:
        restore()


def test_busy_reports_nothing_when_idle():
    res = client.get("/api/busy")
    check("busy is 200", res.status_code == 200)
    body = res.json()
    check("nothing is generating", body["generating"] is False)
    check("nothing is downloading", body["downloading"] is False)


def test_busy_separates_renders_from_downloads():
    # Constructed directly rather than by hitting /api/generate, which would
    # load a 7 GB pipeline. The route only reads Job.kind and Job.done.
    job = app_module.Job("zz-delete-test-job", "generate")
    app_module.JOBS[job.id] = job
    try:
        body = client.get("/api/busy").json()
        check("a live render reads as generating", body["generating"] is True)
        check("a live render is not a download", body["downloading"] is False)
        job.done = True
        check("a finished render reads as idle",
              client.get("/api/busy").json()["generating"] is False)
    finally:
        app_module.JOBS.pop(job.id, None)


if __name__ == "__main__":
    try:
        for test in (
            test_deletes_png_and_sidecar,
            test_missing_name_is_404,
            test_traversal_is_404,
            test_guard_rejects_traversal_directly,
            test_json_name_is_404,
            test_missing_sidecar_still_deletes,
            test_no_token_leaves_every_route_open,
            test_a_gated_server_refuses_a_cookieless_caller,
            test_health_answers_before_the_cookie_exists,
            test_the_token_route_sets_the_cookie_and_redirects,
            test_a_stale_cookie_does_not_shadow_a_fresh_handoff,
            test_a_foreign_origin_is_rejected_even_with_the_cookie,
            test_origin_allowed_is_a_predicate,
            test_query_token_is_scoped_to_the_handoff_route,
            test_the_header_authenticates_without_a_cookie,
            test_busy_reports_nothing_when_idle,
            test_busy_separates_renders_from_downloads,
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
