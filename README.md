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

## Download the app

The quickest way in: one file, no terminal, no Python.

| Platform | File |
|---|---|
| macOS (Apple Silicon) | `.dmg` |
| Windows 10/11 (64-bit) | `.msi` |
| Linux (64-bit) | `.AppImage` or `.deb` |

**[Download the latest release →](https://github.com/giancarlobrusca/local-img/releases/latest)**

On first launch the app downloads its own Python engine — about 1 GB on a Mac,
or up to 7 GB on a PC with an NVIDIA card, where PyTorch ships the much larger
CUDA build — then measures your machine, recommends a model that fits, and
asks before downloading it. Everything after that is offline. Images are
saved to `Pictures/local-img`.

Intel Macs are not supported: without Metal every image would be generated on
the CPU, which the [CPU note](#requirements) below explains is impractical
rather than merely slow.

### The security prompt

The builds are signed ad-hoc but **not notarized** — a Developer ID certificate
costs money per year, and this is a free tool. Both systems will warn you, once:

- **macOS** — "Apple could not verify 'local-img' is free of malware that may
  harm your Mac or compromise your privacy." Click **Done**, then open
  **System Settings → Privacy & Security**, scroll to the line about local-img,
  and click **Open Anyway**. macOS remembers the choice. Control-clicking the
  app and choosing Open used to work instead; macOS 15 removed that shortcut.
- **Windows** — "Windows protected your PC." Click **More info**, then
  **Run anyway**.

If you would rather not click through that, the source install below builds
everything on your own machine.

## Requirements

| | |
|---|---|
| Python | 3.11 or 3.12, **for the source install only** — the app brings its own |
| GPU | Apple Silicon (Metal/MPS) or NVIDIA (CUDA) |
| Memory | A 4 GB GPU or 8 GB of unified memory gets SD 1.5; 16 GB gets every SDXL model; 36 GB+ gets the flux tier |
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
and says so. The NVIDIA scaling factors are still mostly guesses: exactly one
NVIDIA machine has been measured (an RTX 3050 Laptop — see the baselines below),
and it came in about **twice as slow** as `cuda_perf_factor()` predicted for an
unrecognized card. Treat the first-run number on any NVIDIA GPU as optimistic
until your own three renders replace it.

## From source

The repo flow, unchanged: everything the desktop app does, done by hand, on any
machine that already has Python 3.11 or 3.12.

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
./download.sh all               # every model (~104 GB)
./download.sh prune             # drop cached files no pipeline loads
```

### Windows

The shell scripts are macOS/Linux only — a Windows venv puts its interpreter in
`.venv\Scripts\`, not `.venv/bin/`. The equivalent, in PowerShell:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe download.py dreamshaper-8
.venv\Scripts\python.exe app.py                    # → http://127.0.0.1:7788
```

Installing torch from the CUDA index **first** is the step that matters. PyPI's
Windows torch wheel is CPU-only, so a plain `pip install -r requirements.txt`
yields an app that starts, runs, and quietly reports `device: cpu` — correct
images at a fraction of the speed, with no error to tell you why. Doing it in
this order leaves the CUDA build in place, since the later `torch>=2.4` is
already satisfied. Pick the index tag that matches your driver's CUDA version,
which `nvidia-smi` prints in its top-right corner.

Name the 3.12 interpreter explicitly rather than using `py` or `python`: on a
machine where 3.13 is the default, torch has no wheel and the install fails at
resolution time. To force a device, set the variable for the session first:
`$env:LOCAL_IMG_DEVICE = "cpu"`.

## Models

Ten models spanning 4 GB laptop GPUs to 64 GB workstations. Every repo is public and
resolves without a Hugging Face token. The gated repos — FLUX.1-schnell,
FLUX.1-dev, and the Stable Diffusion 3.5 family — are deliberately absent: they
return `GatedRepoError: 401` anonymously, and this app has nowhere to put a token.
The flux tier uses ungated Apache-2.0 derivatives instead.

| Model | Arch | Disk | Needs | Baseline | Notes |
|---|---|---|---|---|---|
| LCM DreamShaper v7 | SD 1.5 | 4.3 GB | 3.5 GB GPU / 8 GB RAM | ~4 s | Latent-consistency distill, 4-8 steps. The fastest route to 768px on a small card. MIT. |
| DreamShaper 8 | SD 1.5 | 2.1 GB | 3 GB GPU / 8 GB RAM | **~18 s** *(measured)* | Smallest download. Broadest concept coverage, huge LoRA ecosystem. |
| SD Turbo | SD 2.1 | 2.6 GB | 3 GB GPU / 8 GB RAM | ~2 s | 1-4 steps at 512px, the smallest footprint here. Base trained on filtered data. |
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

One NVIDIA machine has been measured since — an **RTX 3050 Laptop, 4 GB VRAM**
(Windows 11, driver 577, CUDA 12.9), which is the smallest card the catalog
admits. SD Turbo at 512×512/2 steps → **1.1 s**; DreamShaper 8 at 512×768/25
steps → **12.4 s**; LCM DreamShaper v7 at 768×768/6 steps → **8.4 s**. Peak GPU
memory across the whole card, desktop included, was 2.9 / 3.6 / 3.8 GB: all three
SD 1.5-family entries do clear a 4 GB card, but the 768px one leaves under
200 MB spare, so a second GPU-hungry app open at the same time is the difference
between a render and an out-of-memory error. Against the only baseline that was
itself measured, that card's real scaling factor is about **1.4** — well under
the 3.0 `hardware.py` assumes for a card it does not recognize.

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
recommended defaults, and the image count is capped from the measured memory.
Images are rendered one at a time, so the cap limits how many renders you queue,
not how much memory a render needs. `Cmd+Enter` in the prompt box generates.

The status bar streams live per-step progress over SSE — useful since a 30-step
SDXL render takes a couple of minutes. Generated images land in `outputs/` as PNG
plus a sibling `.json` holding the full parameters (including the seed, so any
image can be reproduced). The filmstrip along the bottom is the local history;
clicking a thumbnail reopens it with its metadata. Deleting removes the PNG and
its `.json` sidecar for good — hover a thumbnail for its `×`, or use the `Delete`
button on the image you are viewing.

Seed `-1` randomizes. Fixing a seed and varying one parameter is the fastest way
to learn what a model responds to.

## Uninstall

An installed local-img is four things in four places, and only one of them is
the app:

| What | Where | Size |
|---|---|---|
| The app | `/Applications`, `Program Files`, or the `.AppImage`/`.deb` | ~10 MB |
| The private Python engine | macOS `~/Library/Application Support/local-img` · Windows `%LOCALAPPDATA%\local-img` · Linux `~/.local/share/local-img` | 1–7 GB |
| Model weights | `~/.cache/huggingface` | 2–34 GB each |
| Your renders | `Pictures/local-img` | whatever you made |

Dragging the app to the Trash removes the first row and leaves the other three.

**From the app.** Open **Storage** in the sidebar. It lists every one of those
with its size and deletes any of them — one model at a time is the everyday
case, and reclaims 7 GB without uninstalling anything. *Delete everything and
uninstall* removes the weights, the Hugging Face dedup cache and the engine in
one go, asking separately about your images, which are the only part that cannot
be downloaded again. The last screen tells you to drag the app itself to the
Trash — on Windows, to remove it in **Apps & features** — because an app cannot
delete itself while it is running.

Nothing here is scheduled or automatic, and nothing outside those four places is
ever touched. The Hugging Face cache is shared with anything else on this machine
that uses it, so the panel offers only the models in this project's catalog: a
repo that belongs to another tool has no name it could be deleted by.

**From the source install.** The same Storage panel is there, without the
uninstall half: the engine is your own `.venv` and is not this project's to
remove. By hand:

```bash
rm -rf ~/.cache/huggingface/hub/models--Lykon--dreamshaper-xl-v2-turbo  # one model
rm -rf ~/.cache/huggingface/xet                # the dedup cache — a copy of the above
rm -rf .venv outputs .local-img                # the engine, the renders, the profile
```

Back up `outputs` first if you want to keep it — of everything that last line
removes, it is the only part nothing can re-download. Delete model directories
by name rather than the whole `hub/` folder, which may hold weights belonging
to something else.

## Layout

```
app.py           FastAPI server — pipeline cache, job queue, SSE progress
models.py        model registry (repo ids, defaults, sizes, fit requirements)
hardware.py      machine detection, per-model fit rules, speed estimates
paths.py         where the profile and the renders live, from two env vars
download.py      resumable weight prefetch
storage.py       what is on disk, by name and size, and how to delete it
delete_test.py   offline checks for the delete route and the session gate
hardware_test.py offline checks for detection, fit, estimates, routes
paths_test.py    offline checks for path resolution in both modes
shell_test.py    offline checks for the port and the parent watchdog
storage_test.py  offline checks for the inventory, its routes, and the panel
web/index.html   the entire UI, no build step
desktop/         the Tauri shell — bootstraps Python, runs app.py, no clone needed
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
  diffusers format (has a `model_index.json`) and is SD 1.5, SDXL, or flux
  architecture.
- **The app and the repo share their weights.** Both use `~/.cache/huggingface`,
  so installing the app after using the source flow re-downloads nothing. The
  app keeps its own hardware profile and puts renders in `Pictures/local-img`.

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
