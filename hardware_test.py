"""Offline checks for hardware detection, fit rules, and speed estimates.

Nothing here loads a model, touches the network, or needs a GPU. Importing app
pulls in torch, which takes a few seconds, but no generation endpoint is called.

Every fixture name carries the zz- prefix and is removed in a finally block, so
real renders in outputs/ and a real .local-img/profile.json are never at risk.

Run: python hardware_test.py
"""

failures: list[str] = []


def check(label: str, condition: bool) -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}")
    if not condition:
        failures.append(label)


# ------------------------------------------------------------------ catalog ---

def test_catalog_columns():
    from models import BY_KEY, DEFAULT_MODEL, MODELS

    for spec in MODELS:
        check(f"{spec.key}: min_budget_gb > 0", spec.min_budget_gb > 0)
        check(f"{spec.key}: min_ram_gb > 0", spec.min_ram_gb > 0)
        check(f"{spec.key}: baseline_seconds > 0", spec.baseline_seconds > 0)
        check(f"{spec.key}: quality_rank > 0", spec.quality_rank > 0)
        check(f"{spec.key}: arch is known", spec.arch in ("sd15", "sdxl", "flux"))
        check(f"{spec.key}: no speed field", not hasattr(spec, "speed"))
        check(f"{spec.key}: no hardcoded recommendation", "recommended" not in spec.tags)

    keys = [m.key for m in MODELS]
    check("no duplicate keys", len(keys) == len(set(keys)))
    check("nine models", len(MODELS) == 9)
    ranks = [m.quality_rank for m in MODELS]
    check("quality ranks are unique", len(ranks) == len(set(ranks)))
    check("default model still exists", DEFAULT_MODEL in BY_KEY)

    for key in ("lcm-dreamshaper-7", "playground-v25", "flex-1-alpha", "shuttle-3-diffusion"):
        check(f"{key} is in the catalog", key in BY_KEY)

    # Distilled and EDM-trained models ship a scheduler that must not be replaced.
    for key in ("sdxl-turbo", "dreamshaper-xl-turbo", "lcm-dreamshaper-7",
                "playground-v25", "flex-1-alpha", "shuttle-3-diffusion"):
        check(f"{key} keeps its scheduler", BY_KEY[key].keep_scheduler)
    for key in ("dreamshaper-8", "juggernaut-xl-v9", "realvis-xl-v4"):
        check(f"{key} takes the DPM++ override", not BY_KEY[key].keep_scheduler)


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
