# local-img

**English** · [Español](README.es.md)

Local text-to-image generation. Prompt in a browser, PNG out. No API keys, no
cloud, no content filtering. After the first weight download it runs fully
offline.

Weights come from the Hugging Face Hub — public repos, no token, no quota — and
every image is generated on your own GPU. Hugging Face is only the file host
here, never an inference API.

![The local-img UI: prompt and model settings on the left, the selected render with its
parameters in the middle, and the local history along the bottom](docs/screenshot.png)

## Requirements

| | |
|---|---|
| Python | 3.11 or 3.12 (PyTorch has no 3.13/3.14 wheels) |
| GPU | Apple Silicon (Metal/MPS) or NVIDIA (CUDA) |
| Memory | 8 GB gets SD 1.5; 16 GB gets every SDXL model; 36 GB+ gets the flux tier |
| Disk | ~2-4 GB per SD 1.5 model, ~7 GB per SDXL model, 26-34 GB per flux model, plus ~2.5 GB of Python deps |

The backend is detected automatically — CUDA, then MPS, then CPU — and shown next
to the title in the UI. Force one with `LOCAL_IMG_DEVICE=cpu ./run.sh`.

**CPU works but is not practical.** There are no fp16 kernels for most of the
UNet, so SDXL falls back to fp32: ~14 GB of RAM and many minutes per image. If
you have no GPU, use LCM DreamShaper or DreamShaper 8 at 512px and expect to wait.

Memory is the binding constraint on a GPU. On first run the app measures your
machine — chip, GPU cores, how much memory Metal or CUDA will actually hand out,
and free disk — and then shows you only the models that fit, with the rest listed
alongside the number that blocks them. Nothing leaves the computer; the result is
saved to `.local-img/profile.json` and can be redone from the sidebar at any time.
You can skip the analysis and see the whole catalog unfiltered.

One pipeline stays resident at a time and the previous one is freed on switch —
two SDXLs will not fit in 16 GB. The flux-architecture models are loaded with
`enable_model_cpu_offload()`, so only one component sits on the GPU at a time;
that is why they need less GPU-visible memory than their download size but a lot
more system RAM.

**Speed estimates start as estimates.** Each model shows a per-image time scaled
from a reference machine (M1 Pro, 16-core GPU). Once you have generated three
images with a model, the app switches to the median of your own recorded times
and says so. The NVIDIA scaling factors are uncalibrated guesses until that
happens — this project has no NVIDIA hardware to measure against.

## Setup

```bash
./setup.sh                      # Python 3.12 venv + torch/diffusers (~2.5 GB, a few minutes)
./download.sh                   # prefetch the default model (~6.9 GB, resumable)
./run.sh                        # → http://127.0.0.1:7788
```

`setup.sh` looks for Python 3.12 or 3.11 specifically, rather than whatever
`python3` points at, because PyTorch still publishes no wheels for 3.13+.

Prefetching is optional but recommended — a dropped connection during a 7 GB pull
otherwise surfaces as a failed prompt. Both paths retry and resume from the HF
cache, so rerunning never restarts from zero.

```bash
./download.sh juggernaut-xl-v9  # a specific model
./download.sh all               # every model (~30 GB)
./download.sh prune             # drop cached files no pipeline loads
```

## Models

Nine models spanning 8 GB laptops to 64 GB workstations. Every repo is public and
resolves without a Hugging Face token. The gated repos — FLUX.1-schnell,
FLUX.1-dev, and the Stable Diffusion 3.5 family — are deliberately absent: they
return `GatedRepoError: 401` anonymously, and this app has nowhere to put a token.
The flux tier uses ungated Apache-2.0 derivatives instead.

| Model | Arch | Disk | Needs | Baseline | Notes |
|---|---|---|---|---|---|
| LCM DreamShaper v7 | SD 1.5 | 4.3 GB | 4 GB GPU / 8 GB RAM | ~4 s | Latent-consistency distill, 4-8 steps. The 8 GB machine's model. MIT. |
| DreamShaper 8 | SD 1.5 | 2.1 GB | 4 GB GPU / 8 GB RAM | **~18 s** *(measured)* | Smallest download. Broadest concept coverage, huge LoRA ecosystem. |
| SDXL Turbo | SDXL | 6.9 GB | 9.5 GB GPU / 16 GB RAM | ~5 s | 3-step previews at 512px. Ignores negative prompts and CFG by design. |
| **DreamShaper XL v2 Turbo** | SDXL | 6.9 GB | 9.5 GB GPU / 16 GB RAM | **~40 s** *(measured)* | Best quality-per-second. SDXL at 1024px in 7 steps. |
| Playground v2.5 | SDXL | 7.0 GB | 9.5 GB GPU / 16 GB RAM | ~200 s | Trained from scratch for aesthetics. Ships an EDM scheduler it needs. |
| Juggernaut XL v9 | SDXL | 7.1 GB | 9.5 GB GPU / 16 GB RAM | ~180 s | Photorealism specialist. Full 30-step sampling. |
| RealVisXL V4.0 | SDXL | 6.9 GB | 9.5 GB GPU / 16 GB RAM | ~180 s | The other photoreal SDXL fine-tune; stronger on people. |
| Flex.1 alpha | flux, 8B | 26.3 GB | 20 GB GPU / 36 GB RAM | ~300 s | Apache-2.0 schnell derivative with a trained guidance embedder. |
| Shuttle 3 Diffusion | flux, 12B | 33.7 GB | 26 GB GPU / 48 GB RAM | ~120 s | Apache-2.0 FLUX.1-schnell derivative. Best prompt adherence, 4 steps. |

There is no fixed default. The app recommends the highest-quality model that fits
the machine it is running on — DreamShaper XL v2 Turbo on a 16 GB M1 Pro,
Shuttle 3 Diffusion on a 64 GB M4 Max, DreamShaper 8 on an 8 GB Mac.

Baselines are per image on an **M1 Pro, 16 GB unified memory** — a reference
point, not a spec, and scaled to your machine in the UI. Measured there:
DreamShaper 8 at 512×512/20 steps → **18.8 s**; DreamShaper XL Turbo at
1024×1024/7 steps → **39.5 s** and **42.8 s**. Everything else is extrapolated
from those per-step costs. The flux numbers are the least certain of all: no flux
model fits in this machine's 11.8 GB of usable memory, so that path is untested
here and its memory requirements are conservative estimates.

Disk sizes are what the pipeline actually loads. The repos themselves are much
larger — dreamshaper-xl-v2-turbo is 41.6 GB in full, because it also ships three
standalone single-file checkpoints and fp32 copies of every component that the
fp16 pipeline never touches. `download.sh` fetches only the needed files, and
`./download.sh prune` reclaims the rest if a broader tool already pulled them.

### On content filtering

Nothing in this stack filters prompts or images:

- SDXL pipelines define no safety checker at all.
- The SD 1.5 pipeline is built with `safety_checker=None, requires_safety_checker=False`
  (`app.py`), so no output is blurred or replaced.
- There is no prompt-side keyword matching anywhere in this codebase.

The community fine-tunes (DreamShaper, Juggernaut, RealVis) were also trained
without content filtering, so the restriction isn't merely disabled at runtime —
it isn't in the weights either. Worth knowing: this is a property of *the weights*,
not a switch. FLUX.1-dev, by contrast, was trained on curated data, so it can't
produce what it never saw regardless of pipeline settings — the flux-architecture
models here are schnell derivatives, which are less curated but not uncurated.

You own the output and the responsibility for it. Local generation removes the
provider's guardrails, not the law — non-consensual imagery of real people and
sexual content involving minors are crimes in essentially every jurisdiction, and
no amount of "it ran on my laptop" changes that.

## Using it

On first run a short wizard measures the machine and shows you the catalog split
into what it recommends, what also works, and what will not fit and why. After
that, the sidebar carries a one-line hardware summary; click it for the full
profile, a *Re-analyze* button, and a checkbox that unhides the models that do
not fit.

Left panel: prompt, model, negative prompt, and a Settings drawer for steps,
guidance, dimensions, image count, and seed. Selecting a model loads its
recommended defaults, and the image count caps at what the measured memory
supports. `Cmd+Enter` in the prompt box generates.

The status bar streams live per-step progress over SSE — useful since a 30-step
SDXL render takes a couple of minutes. Generated images land in `outputs/` as PNG
plus a sibling `.json` holding the full parameters (including the seed, so any
image can be reproduced). The filmstrip along the bottom is the local history;
clicking a thumbnail reopens it with its metadata. Deleting removes the PNG and
its `.json` sidecar for good — hover a thumbnail for its `×`, or use the `Delete`
button on the image you are viewing.

Seed `-1` randomizes. Fixing a seed and varying one parameter is the fastest way
to learn what a model responds to.

## Layout

```
app.py           FastAPI server — pipeline cache, job queue, SSE progress
models.py        model registry (repo ids, defaults, sizes, fit requirements)
hardware.py      machine detection, per-model fit rules, speed estimates
download.py      resumable weight prefetch
delete_test.py   offline checks for the delete route (no GPU work)
hardware_test.py offline checks for detection, fit, estimates, routes (no GPU work)
web/index.html   the entire UI, no build step
outputs/         generated PNGs + parameter sidecars (gitignored)
.local-img/      the detected hardware profile (gitignored)
```

## Notes

- **First generation on a model is slow** — it downloads weights, then loads ~7 GB
  into memory. Subsequent prompts reuse the resident pipeline.
- **Switching models** unloads the previous one and reloads from disk (~15–30 s).
  Batch your work by model.
- **Memory pressure**: close browser-heavy sessions before 1024px SDXL runs. On
  unified memory especially, once the OS starts swapping, generation time doubles.
- **Non-square sizes** must be multiples of 8 (the app rounds down). SDXL was
  trained at ~1 megapixel — 1024×1024, 1152×896, 896×1152 behave best; pushing
  past ~1536 on either axis produces duplicated subjects and risks OOM.
- Adding a model is a new entry in `models.py`, as long as the repo is in
  diffusers format (has a `model_index.json`) and is SD 1.5 or SDXL architecture.

## Don't deploy this

Run it on your own machine. It is built for exactly one user on localhost, and
hosting it publicly breaks in several directions at once:

- The server binds to `127.0.0.1` and there is **no auth and no rate limiting**.
  An open endpoint that burns 40 s of GPU per request is a free denial-of-service.
- `outputs/` is **global**. The gallery shows every visitor everyone else's
  images, and the delete route lets anyone remove anyone's files.
- A single `GEN_LOCK` serializes all generation, so users queue behind each other.
- Serverless hosts are out entirely: no persistent disk for 7 GB of weights,
  plus size caps and request timeouts measured in seconds.
- Anything without a GPU falls back to CPU and grinds, as described above.

Making it multi-tenant means real queueing, per-session storage, auth and quotas
— a different project. And a public, anonymous, unfiltered image generator
carries a very different legal exposure than the same tool on your own laptop.

## License

MIT — see [LICENSE](LICENSE). The model weights are **not** covered by it; each
one carries its own license on its Hugging Face repo (the Stability models ship
under the CreativeML OpenRAIL++-M license, which restricts some uses). Check the
one you plan to use, especially commercially.
