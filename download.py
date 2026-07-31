"""Prefetch model weights into the Hugging Face cache, with resume on failure.

Multi-GB pulls routinely drop mid-transfer. Downloading up front — rather than
lazily inside a generation request — means a dropped connection costs a retry
instead of a failed prompt.

Fetching is delegated to `DiffusionPipeline.download`, which pulls *only* the
files the pipeline actually loads. This matters: these repos also ship
single-file checkpoints and fp32 copies of every component. Lykon/dreamshaper-xl-v2-turbo
is 41.6 GB in full, but the fp16 diffusers pipeline needs just 6.9 GB of it.

    ./download.sh                     # the default model
    ./download.sh juggernaut-xl-v9    # a specific one
    ./download.sh all                 # every model
    ./download.sh prune               # delete cached files no pipeline loads
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

from models import BY_KEY, DEFAULT_MODEL, MODELS

HUB = Path.home() / ".cache" / "huggingface" / "hub"


def _pipeline_cls(spec):
    from diffusers import (
        FluxPipeline,
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
    )

    return {
        "flux": FluxPipeline,
        "sdxl": StableDiffusionXLPipeline,
    }.get(spec.arch, StableDiffusionPipeline)


def fetch(spec, attempts=6):
    """Download one model. Returns the snapshot folder, or None on failure."""
    cls = _pipeline_cls(spec)
    dtype = torch.bfloat16 if spec.arch == "flux" else torch.float16
    print(f"\n=== {spec.name}  ({spec.repo}, ~{spec.download_gb} GB)")
    for attempt in range(1, attempts + 1):
        try:
            try:
                folder = cls.download(
                    spec.repo, variant="fp16", use_safetensors=True,
                    torch_dtype=dtype,
                )
            except (ValueError, OSError, EnvironmentError):
                # Not every repo publishes an fp16 variant of every component.
                folder = cls.download(
                    spec.repo, use_safetensors=True, torch_dtype=dtype,
                )
            print(f"--- {spec.name}: complete")
            return Path(folder)
        except Exception as exc:
            print(f"!!! attempt {attempt}/{attempts}: {type(exc).__name__}: {exc}")
            if attempt == attempts:
                print(f"--- {spec.name}: GAVE UP — rerun to resume from the cache")
                return None
            time.sleep(min(30, 3 * attempt))


def repo_dir(repo: str) -> Path:
    return HUB / ("models--" + repo.replace("/", "--"))


WEIGHT_SUFFIXES = (".safetensors", ".bin", ".ckpt", ".pt", ".onnx")


def needed_files(snapshot: Path) -> set[Path]:
    """The files a diffusers pipeline actually loads from a snapshot.

    `DiffusionPipeline.download` returns the whole snapshot directory, which by
    then also holds anything previously fetched — so the needed set has to be
    derived from `model_index.json` instead of from what's on disk.
    """
    index = snapshot / "model_index.json"
    if not index.exists():
        return set()

    keep = {index}
    spec = json.loads(index.read_text(encoding="utf-8"))
    components = [
        name for name, value in spec.items()
        if not name.startswith("_") and isinstance(value, list) and (snapshot / name).is_dir()
        # We load with safety_checker=None, so its weights are never read.
        and name not in ("safety_checker", "feature_extractor")
    ]

    # from_pretrained(variant="fp16") is all-or-nothing: it needs the fp16 file
    # for every weight-bearing component, or it falls back to plain for all.
    weighted = []
    for name in components:
        files = [p for p in (snapshot / name).iterdir() if p.suffix in WEIGHT_SUFFIXES]
        if files:
            weighted.append((name, files))
    use_fp16 = all(
        any(".fp16" in p.name for p in files) for _, files in weighted
    ) and bool(weighted)

    for name in components:
        for p in (snapshot / name).iterdir():
            if p.suffix not in WEIGHT_SUFFIXES:
                keep.add(p)          # configs, tokenizer vocab/merges, sentencepiece
            elif p.suffix != ".safetensors":
                continue             # never load .bin/.ckpt when safetensors exist
            elif (".fp16" in p.name) == use_fp16:
                keep.add(p)
    return keep


def prune():
    """Drop cached blobs that no configured pipeline loads.

    These repos also ship standalone single-file checkpoints and fp32 copies of
    every component — for dreamshaper-xl-v2-turbo that's 41.6 GB on disk where
    the fp16 pipeline needs 6.9 GB.
    """
    total = 0
    for spec in MODELS:
        rd = repo_dir(spec.repo)
        if not rd.exists():
            continue

        keep_blobs = set()
        snapshots = list((rd / "snapshots").iterdir()) if (rd / "snapshots").exists() else []
        for snap in snapshots:
            for p in needed_files(snap):
                if p.exists():
                    keep_blobs.add(os.path.realpath(p))

        if not keep_blobs:
            print(f"  skip {spec.name}: no complete snapshot found")
            continue

        freed = 0
        for snap in (rd / "snapshots").rglob("*"):
            if (snap.is_file() or snap.is_symlink()) and os.path.realpath(snap) not in keep_blobs:
                snap.unlink()
        for blob in (rd / "blobs").glob("*"):
            if os.path.realpath(blob) not in keep_blobs:
                freed += blob.stat().st_size
                blob.unlink()
        # Xet dedup metadata for blobs we just dropped.
        shutil.rmtree(rd / "trees", ignore_errors=True)

        total += freed
        print(f"  {spec.name}: freed {freed / 1e9:.1f} GB")

    # Xet keeps a separate chunk cache that is pure duplicate of the blobs.
    xet = Path.home() / ".cache" / "huggingface" / "xet"
    if xet.exists():
        size = sum(f.stat().st_size for f in xet.rglob("*") if f.is_file())
        shutil.rmtree(xet, ignore_errors=True)
        total += size
        print(f"  xet chunk cache: freed {size / 1e9:.1f} GB")

    print(f"\n=== reclaimed {total / 1e9:.1f} GB")


def is_cached(spec) -> bool:
    """Whether this model's weights are already on disk.

    A snapshot counts only when every file the pipeline loads is present, so a
    half-finished download does not read as ready.
    """
    snapshots = repo_dir(spec.repo) / "snapshots"
    if not snapshots.is_dir():
        return False
    for snap in snapshots.iterdir():
        files = needed_files(snap)
        if files and all(p.exists() for p in files):
            return True
    return False


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL

    if arg == "prune":
        prune()
        sys.exit(0)
    if arg == "all":
        targets = MODELS
    elif arg in BY_KEY:
        targets = [BY_KEY[arg]]
    else:
        print(f"unknown model '{arg}'. Options: {', '.join(BY_KEY)}, all, prune")
        sys.exit(1)

    results = [(s.name, fetch(s) is not None) for s in targets]
    print("\n=== summary")
    for name, ok in results:
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}")
    sys.exit(0 if all(ok for _, ok in results) else 1)
