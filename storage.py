"""What local-img has put on this disk, as an inventory.

Composes numbers and nothing else. There is no FastAPI here, no knowledge of
the desktop shell, and no deletion of the runtime this module measures — Python
runs from inside that runtime, and Windows does not let a process unlink files
it holds open. The shell takes that step.

The central guarantee is negative and structural. The model list is built by
iterating `models.MODELS`, never by listing the Hugging Face cache. That cache
is shared with the repo flow on purpose (`paths.hf_cache_dir()` says so) and may
hold weights belonging to other tools; a repo that is not in the catalog cannot
appear in this inventory, and therefore cannot be named to a delete route. That
is a property of the loop, not a rule someone has to remember.
"""

from __future__ import annotations

from pathlib import Path

import download
import paths
from models import MODELS


def runtime_dir() -> Path:
    """The private Python runtime the desktop shell installs under the data
    directory. Measured here, deleted nowhere — see the module docstring."""
    return paths.data_dir() / "runtime"


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _complete(spec) -> bool:
    """is_cached, defensively.

    is_cached walks the cache; one unreadable directory or one corrupt
    model_index.json in there must not turn the whole panel into a 500. Unknown
    means incomplete, which is the answer that offers the bytes for deletion.
    """
    try:
        return download.is_cached(spec)
    except (OSError, ValueError):
        return False


def outputs_summary() -> dict:
    """Renders and their parameter sidecars: how many, how much, and where."""
    out = paths.outputs_dir()
    if not out.is_dir():
        return {"count": 0, "bytes": 0, "path": str(out)}
    count = 0
    total = 0
    for png in out.glob("*.png"):
        count += 1
        total += _size(png) + _size(png.with_suffix(".json"))
    return {"count": count, "bytes": total, "path": str(out)}


def rest_bytes() -> int:
    """The data directory minus the runtime: logs, profile.json, the stamp, a
    leftover archive. Reported as one line because none of it is individually
    interesting and all of it goes together."""
    return download.dir_size(paths.data_dir()) - download.dir_size(runtime_dir())


def inventory() -> dict:
    """
    runtime:  int                              # data_dir()/runtime
    models:   [{key, name, bytes, complete}]   # catalog models present in the cache
    xet:      int                              # dedup cache, not attributable to one model
    outputs:  {count, bytes, path}
    rest:     int                              # logs, profile.json, leftovers
    """
    models = []
    for spec in MODELS:
        size = download.size_on_disk(spec)
        if not size:
            continue        # nothing on disk: nothing to show, nothing to free
        models.append({
            "key": spec.key,
            "name": spec.name,
            "bytes": size,
            "complete": _complete(spec),
        })
    return {
        "runtime": download.dir_size(runtime_dir()),
        "models": models,
        "xet": download.dir_size(download.xet_dir()),
        "outputs": outputs_summary(),
        "rest": rest_bytes(),
    }


def remove_outputs() -> tuple[int, list[str]]:
    """Every render and its sidecar. Returns bytes freed and paths that resisted.

    Scoped to `*.png` and the `.json` beside each one, exactly like the existing
    single-image delete route. A file the user dropped into that folder is not
    this project's to remove.
    """
    out = paths.outputs_dir()
    if not out.is_dir():
        return 0, []
    freed = 0
    resisted: list[str] = []
    for png in sorted(out.glob("*.png")):
        for path in (png, png.with_suffix(".json")):
            if not path.exists():
                continue
            size = _size(path)
            try:
                path.unlink()
                freed += size
            except OSError:
                resisted.append(str(path))
    return freed, resisted


def remove_xet() -> tuple[int, list[str]]:
    """The Xet chunk cache.

    A dedup store that duplicates bytes already in `blobs/` — `download.prune()`
    has said so for as long as it has existed. Deleting it loses no weights; at
    worst a later download re-chunks what it could have reused.
    """
    return download.remove_tree(download.xet_dir())
