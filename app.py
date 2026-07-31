"""Local text-to-image server. Runs on Apple Metal (MPS), NVIDIA (CUDA), or CPU.

Runs entirely offline after the model weights are cached. Start with ./run.sh
then open http://127.0.0.1:7788

Bound to localhost on purpose. This serves an unauthenticated, unfiltered
generation endpoint with a shared gallery and a delete route — it is a
single-user tool, not something to expose to a network.
"""

from __future__ import annotations

import gc
import json
import os
import queue
import random
import shutil
import threading
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import download
import hardware
from models import BY_KEY, DEFAULT_MODEL, MODELS

ROOT = Path(__file__).parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

def _pick_device() -> str:
    """CUDA, then Apple Metal, then CPU. Override with LOCAL_IMG_DEVICE."""
    forced = os.environ.get("LOCAL_IMG_DEVICE", "").strip()
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _pick_device()
# Both GPU backends run fp16. CPU lacks fp16 kernels for much of the UNet, so it
# gets fp32 — twice the memory (~14 GB for SDXL) and still minutes per image.
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16

app = FastAPI(title="local-img")


# ---------------------------------------------------------------- pipeline ---

def pipeline_plan(spec) -> dict:
    """How this model is loaded, as data — so the decision is testable without
    a GPU, a download, or an import of diffusers.

    flux pipelines are bf16 (the published weights already are) and CPU-offloaded
    rather than moved onto the device: only one component is resident on the GPU
    at a time, which is why min_budget_gb is far below the download size while
    min_ram_gb is far above it.
    """
    if spec.arch == "flux":
        return {
            "pipeline": "FluxPipeline",
            "dtype": torch.bfloat16,
            "offload": True,
            "override_scheduler": False,
            "disable_safety": False,
        }
    return {
        "pipeline": "StableDiffusionXLPipeline" if spec.arch == "sdxl"
                    else "StableDiffusionPipeline",
        "dtype": DTYPE,
        "offload": False,
        # Distilled and EDM-trained models need their shipped scheduler;
        # everything else converges much better on DPM++ SDE Karras.
        "override_scheduler": not spec.keep_scheduler,
        # No content filter. Nothing is blurred or replaced on the way out.
        "disable_safety": spec.arch == "sd15",
    }


class PipelineCache:
    """Holds at most one pipeline in memory — 16 GB unified memory can't hold two SDXLs."""

    def __init__(self):
        self.key: str | None = None
        self.pipe = None
        self.lock = threading.Lock()

    def get(self, key: str, on_status):
        if self.key == key and self.pipe is not None:
            return self.pipe

        if self.pipe is not None:
            on_status(f"unloading {self.key}")
            del self.pipe
            self.pipe = None
            self.key = None
            gc.collect()
            if DEVICE == "mps":
                torch.mps.empty_cache()
            elif DEVICE == "cuda":
                torch.cuda.empty_cache()

        spec = BY_KEY[key]
        on_status(f"loading {spec.name} (first run downloads ~{spec.download_gb} GB)")

        plan = pipeline_plan(spec)

        from diffusers import DPMSolverMultistepScheduler
        import diffusers

        cls = getattr(diffusers, plan["pipeline"])
        kwargs = dict(torch_dtype=plan["dtype"], use_safetensors=True)
        if plan["disable_safety"]:
            kwargs.update(safety_checker=None, requires_safety_checker=False)

        # Multi-GB downloads over a flaky link drop mid-transfer; already-fetched
        # shards stay in the HF cache, so retrying resumes rather than restarts.
        last = None
        for attempt in range(4):
            try:
                try:
                    pipe = cls.from_pretrained(spec.repo, variant="fp16", **kwargs)
                except (ValueError, OSError, EnvironmentError):
                    # Neither flux repo publishes an fp16 variant — those weights
                    # are already bf16 — and not every repo ships one per component.
                    pipe = cls.from_pretrained(spec.repo, **kwargs)
                break
            except Exception as exc:
                last = exc
                if attempt == 3:
                    raise RuntimeError(
                        f"could not fetch {spec.repo} after 4 attempts ({exc}). "
                        f"Run ./download.sh {spec.key} to prefetch the weights."
                    ) from exc
                on_status(f"download interrupted, resuming (attempt {attempt + 2}/4)")
                time.sleep(2 + 3 * attempt)

        if plan["override_scheduler"]:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                algorithm_type="sde-dpmsolver++",
                use_karras_sigmas=True,
            )

        pipe.set_progress_bar_config(disable=True)
        if plan["offload"]:
            # One component on the GPU at a time; the rest waits in system RAM.
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(DEVICE)
            pipe.enable_attention_slicing()
            pipe.vae.enable_slicing()  # keeps VAE decode of a 1024px latent inside 16 GB

        self.pipe, self.key = pipe, key
        return pipe


CACHE = PipelineCache()


# -------------------------------------------------------------------- jobs ---

class Job:
    def __init__(self, job_id: str):
        self.id = job_id
        self.events: queue.Queue = queue.Queue()
        self.done = False

    def emit(self, **payload):
        self.events.put(payload)


JOBS: dict[str, Job] = {}
GEN_LOCK = threading.Lock()


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    model: str = DEFAULT_MODEL
    steps: int = Field(default=0, ge=0, le=100)
    guidance: float = Field(default=-1.0, ge=-1.0, le=20.0)
    width: int = Field(default=0, ge=0, le=1536)
    height: int = Field(default=0, ge=0, le=1536)
    seed: int = -1
    batch: int = Field(default=1, ge=1, le=4)


_MEMORY_SIGNS = ("out of memory", "outofmemory", "cannot allocate",
                 "insufficient memory", "failed to allocate")


def memory_hint(exc: Exception, spec) -> str:
    """A suggestion appended to a failure that looks like memory exhaustion.

    Note the real limit: on MPS an out-of-memory condition sometimes hangs or
    kills the process instead of raising cleanly. That is precisely why the
    default hides non-fitting models rather than merely warning about them —
    this message only helps in the cases that do raise.
    """
    if spec is None:
        return ""
    text = f"{type(exc).__name__}: {exc}".lower()
    if not any(sign in text for sign in _MEMORY_SIGNS):
        return ""
    return (f" — {spec.name} did not fit in memory. Pick a smaller model; "
            f"the hardware panel in the sidebar lists what this machine can run.")


def run_job(job: Job, req: GenerateRequest):
    try:
        spec = BY_KEY[req.model]
        steps = req.steps or spec.steps
        guidance = spec.guidance if req.guidance < 0 else req.guidance
        width = req.width or spec.width
        height = req.height or spec.height
        # SD/SDXL UNets require dimensions divisible by 8.
        width, height = (width // 8) * 8, (height // 8) * 8

        with GEN_LOCK:
            job.emit(stage="load", message="preparing model")
            pipe = CACHE.get(req.model, lambda m: job.emit(stage="load", message=m))

            for i in range(req.batch):
                seed = random.randint(0, 2**31 - 1) if req.seed < 0 else req.seed + i
                generator = torch.Generator("cpu").manual_seed(seed)
                started = time.time()

                def on_step(_pipe, step, _t, kwargs, _n=i):
                    job.emit(
                        stage="step",
                        step=step + 1,
                        total=steps,
                        index=_n,
                        batch=req.batch,
                        elapsed=round(time.time() - started, 1),
                    )
                    return kwargs

                call = dict(
                    prompt=req.prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    width=width,
                    height=height,
                    generator=generator,
                    callback_on_step_end=on_step,
                )
                if spec.supports_negative and req.negative_prompt.strip():
                    call["negative_prompt"] = req.negative_prompt

                image = pipe(**call).images[0]

                stamp = time.strftime("%Y%m%d-%H%M%S")
                name = f"{stamp}-{req.model}-{seed}.png"
                path = OUTPUTS / name
                image.save(path)
                meta = {
                    "prompt": req.prompt,
                    "negative_prompt": req.negative_prompt,
                    "model": req.model,
                    "steps": steps,
                    "guidance": guidance,
                    "size": [width, height],
                    "seed": seed,
                    "seconds": round(time.time() - started, 1),
                }
                path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                job.emit(stage="image", url=f"/outputs/{name}", meta=meta)

        job.emit(stage="done")
    except Exception as exc:  # surfaced verbatim in the UI
        hint = memory_hint(exc, BY_KEY.get(req.model))
        job.emit(stage="error", message=f"{type(exc).__name__}: {exc}{hint}")
    finally:
        job.done = True


# ------------------------------------------------------------------ routes ---

def _ready(spec) -> bool:
    """Whether the weights are cached, defensively.

    is_cached walks the Hugging Face cache; one unreadable directory or one
    corrupt model_index.json in there must not turn the whole catalog route into
    a 500 and leave the page blank. Unknown means not ready.
    """
    try:
        return download.is_cached(spec)
    except (OSError, ValueError):
        return False


def catalog_payload():
    """The catalog as the browser sees it: specs plus fit, estimate, readiness.

    app.py computes nothing here — hardware.py decides what fits and how long it
    takes, and this function only assembles the answer.
    """
    profile = hardware.load()
    fits = hardware.fit_all(profile, MODELS)
    history = hardware.read_history(OUTPUTS)

    models = []
    for spec in MODELS:
        fit = fits[spec.key]
        est = hardware.estimate(profile, spec, history.get(spec.key, []))
        models.append({
            **spec.to_json(),
            "fits": fit.fits,
            "reason": fit.reason,
            "max_batch": fit.max_batch,
            "recommended": fit.recommended,
            "estimate": asdict(est),
            "ready": _ready(spec),
        })

    picked = next((m["key"] for m in models if m["recommended"]), None)
    if picked is None:
        # Nothing is recommended: either there is no profile to recommend from
        # (the hardcoded default is right), or the machine cannot run anything —
        # in which case offering the default would offer a model this very
        # payload marks fits: false. Offer the cheapest entry instead.
        picked = (DEFAULT_MODEL if any(m["fits"] for m in models)
                  else hardware.least_demanding(MODELS).key)
    return {
        "device": DEVICE,
        "default": picked,
        "profile": asdict(profile) if profile else None,
        "models": models,
        "cached": CACHE.key,     # the pipeline resident in memory, if any
    }


class ScanRequest(BaseModel):
    skip: bool = False


@app.get("/api/models")
def list_models():
    return catalog_payload()


@app.post("/api/hardware/scan")
def scan_hardware(req: ScanRequest = ScanRequest()):
    """Measure the machine and persist the result — or persist a declined scan.

    A skipped profile is a record of a choice, not a measurement: it shows the
    full catalog, recommends nothing, and leaves the offer to analyze standing.
    """
    profile = hardware.skipped_profile(DEVICE) if req.skip else hardware.detect(DEVICE)
    hardware.save(profile)
    return catalog_payload()


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if req.model not in BY_KEY:
        raise HTTPException(400, f"unknown model {req.model}")
    if not req.prompt.strip():
        raise HTTPException(400, "prompt is empty")
    job_id = f"{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
    job = Job(job_id)
    JOBS[job_id] = job
    threading.Thread(target=run_job, args=(job, req), daemon=True).start()
    return {"job": job_id}


# 15% over the pipeline size: the HF cache briefly holds both an incoming blob
# and its snapshot entry, and Xet keeps a chunk store alongside.
DISK_HEADROOM = 1.15
POLL_SECONDS = 0.5


def disk_shortfall(spec, free_gb: float) -> str | None:
    """The refusal message when a download cannot fit, or None when it can."""
    need = spec.download_gb * DISK_HEADROOM
    if free_gb >= need:
        return None
    return (f"not enough disk: {spec.name} needs {need:.1f} GB "
            f"and only {free_gb:.1f} GB is free")


def _watch_download(job: Job, spec, stop: threading.Event):
    """Report progress by sampling the size of the repo's blob directory.

    Approximate on purpose, and correct across transfer backends: Xet's chunk
    store makes a per-file byte counter meaningless, but bytes landing in
    `blobs/` are bytes landing in `blobs/` either way.
    """
    blobs = download.repo_dir(spec.repo) / "blobs"
    while not stop.wait(POLL_SECONDS):
        try:
            done = sum(f.stat().st_size for f in blobs.glob("*") if f.is_file())
        except OSError:
            continue
        gb_done = done / 1e9
        if stop.is_set():
            # The scan above (glob + stat over every blob) takes real I/O time,
            # so run_download's finally may have set stop and already emitted
            # "done" while we were scanning. Recheck right before emitting so a
            # stray "download" event can't land after the job is over — do not
            # remove this as a duplicate of the loop's stop.wait() check above.
            break
        job.emit(
            stage="download",
            pct=min(99, round(gb_done / spec.download_gb * 100)),
            gb_done=round(gb_done, 2),
            gb_total=spec.download_gb,
        )


def run_download(job: Job, spec):
    stop = threading.Event()
    try:
        free_gb = shutil.disk_usage(ROOT).free / 1e9
        problem = disk_shortfall(spec, free_gb)
        if problem:
            # Fail before transferring anything rather than filling the disk.
            job.emit(stage="error", message=problem)
            return

        job.emit(stage="download", pct=0, gb_done=0.0, gb_total=spec.download_gb)
        threading.Thread(target=_watch_download, args=(job, spec, stop), daemon=True).start()
        # download.fetch already retries six times with backoff and resumes from
        # the cache — no retry logic is duplicated here.
        # fetch() reports a total failure by returning None rather than raising
        # — its __main__ caller depends on that — so the None has to be checked
        # here. Without it a failed download still drove the bar to 100%.
        if download.fetch(spec) is None:
            job.emit(stage="error", message=(
                f"{spec.name} was not downloaded — every attempt to fetch the "
                f"weights failed (the server console has the reason). Trying "
                f"again resumes from whatever is already in the cache."
            ))
            return
        job.emit(stage="download", pct=100, gb_done=spec.download_gb, gb_total=spec.download_gb)
        job.emit(stage="done")
    except Exception as exc:  # surfaced verbatim in the UI
        job.emit(stage="error", message=f"{type(exc).__name__}: {exc}")
    finally:
        stop.set()
        job.done = True


@app.post("/api/download/{key}")
def start_download(key: str):
    if key not in BY_KEY:
        raise HTTPException(400, f"unknown model {key}")
    job_id = f"{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
    job = Job(job_id)
    JOBS[job_id] = job
    threading.Thread(target=run_download, args=(job, BY_KEY[key]), daemon=True).start()
    return {"job": job_id}


@app.get("/api/stream/{job_id}")
def stream(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")

    def events():
        while True:
            try:
                payload = job.events.get(timeout=1.0)
            except queue.Empty:
                if job.done and job.events.empty():
                    break
                yield ": keepalive\n\n"
                continue
            yield f"data: {json.dumps(payload)}\n\n"
            if payload.get("stage") in ("done", "error"):
                break
        JOBS.pop(job_id, None)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/gallery")
def gallery(limit: int = 40):
    pngs = sorted(OUTPUTS.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    items = []
    for p in pngs[:limit]:
        meta_path = p.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        items.append({"url": f"/outputs/{p.name}", "meta": meta})
    return {"items": items}


@app.get("/", response_class=HTMLResponse)
def index():
    # encoding="utf-8" is not optional. Without it Python decodes with the
    # platform's locale encoding — cp1252 on a Windows machine — and every
    # typographic character in the page (·, —) reaches the browser as mojibake,
    # or the route raises outright on a strictly ASCII locale.
    return (ROOT / "web" / "index.html").read_text(encoding="utf-8")


@app.get("/outputs/{name}")
def output(name: str):
    path = (OUTPUTS / name).resolve()
    if not str(path).startswith(str(OUTPUTS.resolve())) or not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


@app.delete("/api/outputs/{name}")
def delete_output(name: str):
    path = (OUTPUTS / name).resolve()
    # Exact parent match, not a string prefix: a sibling outputs-old/ must not
    # pass, and only a render (plus its sidecar) may ever be unlinked.
    if path.parent != OUTPUTS.resolve() or path.suffix != ".png" or not path.exists():
        raise HTTPException(404)
    path.unlink()
    path.with_suffix(".json").unlink(missing_ok=True)
    return {"deleted": name}


app.mount("/web", StaticFiles(directory=ROOT / "web"), name="web")


if __name__ == "__main__":
    print(f"\n  local-img — device: {DEVICE}, dtype: {DTYPE}")
    if DEVICE == "cpu":
        print("  WARNING: no GPU detected. Expect many minutes per image, and")
        print("           ~14 GB of RAM for the SDXL models. Try dreamshaper-8.")
    print("  open http://127.0.0.1:7788\n")
    uvicorn.run(app, host="127.0.0.1", port=7788, log_level="warning")
