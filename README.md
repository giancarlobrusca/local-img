# local-img

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
| Memory | 16 GB unified, or 8 GB VRAM, for the SDXL models; roughly half that for SD 1.5 |
| Disk | ~2 GB per SD 1.5 model, ~7 GB per SDXL model, plus ~2.5 GB of Python deps |

The backend is detected automatically — CUDA, then MPS, then CPU — and shown next
to the title in the UI. Force one with `LOCAL_IMG_DEVICE=cpu ./run.sh`.

**CPU works but is not practical.** There are no fp16 kernels for most of the
UNet, so SDXL falls back to fp32: ~14 GB of RAM and many minutes per image. If
you have no GPU, use DreamShaper 8 at 512px and expect to wait.

Memory is the binding constraint on a GPU. SDXL at fp16 is ~7 GB of weights plus
activations, so the app keeps **one** pipeline resident at a time and frees the
previous one on switch — two SDXLs will not fit in 16 GB. FLUX (12B) is
deliberately not included: it needs 4-bit quantization to fit on this class of
hardware and still takes several minutes per image.

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

Picked to fit a 16 GB laptop-class GPU — everything here stays in memory and
finishes in a workable amount of time.

| Model | Disk | Speed | Notes |
|---|---|---|---|
| **DreamShaper XL v2 Turbo** *(default)* | 6.9 GB | **~40 s** *(measured)* | Best quality/speed balance. SDXL at 1024px in 7 steps. |
| DreamShaper 8 (SD 1.5) | 2.1 GB | **~18 s** *(measured)* | Smallest and fastest. Broadest concept coverage, huge LoRA ecosystem. |
| Juggernaut XL v9 | 7.1 GB | ~3 min *(estimated)* | Photorealism specialist. Full 30-step sampling. |
| RealVisXL V4.0 | 6.9 GB | ~3 min *(estimated)* | The other photoreal SDXL fine-tune; stronger on people. |
| SDXL Turbo | 6.9 GB | ~5 s *(estimated)* | 3-step previews at 512px. Ignores negative prompts and CFG by design. |

Timings are from an **M1 Pro, 16 GB unified memory** — treat them as a reference
point, not a spec. Measured there: DreamShaper 8 at 512×512/20 steps → **18.8 s**;
DreamShaper XL Turbo at 1024×1024/7 steps → **39.5 s** and **42.8 s**. The two
photoreal models are extrapolated from the XL Turbo per-step cost, so those are
estimates. A recent NVIDIA card is considerably faster across the board.

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
produce what it never saw regardless of pipeline settings.

You own the output and the responsibility for it. Local generation removes the
provider's guardrails, not the law — non-consensual imagery of real people and
sexual content involving minors are crimes in essentially every jurisdiction, and
no amount of "it ran on my laptop" changes that.

## Using it

Left panel: prompt, model, negative prompt, and a Settings drawer for steps,
guidance, dimensions, image count, and seed. Selecting a model loads its
recommended defaults. `Cmd+Enter` in the prompt box generates.

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
models.py        model registry (repo ids, defaults, sizes)
download.py      resumable weight prefetch
delete_test.py   offline checks for the delete route (no GPU work)
web/index.html   the entire UI, no build step
outputs/         generated PNGs + parameter sidecars (gitignored)
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
