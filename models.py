"""Model registry for local image generation.

Every entry runs fully offline after the first download, and every repo resolves
anonymously — public weights, no Hugging Face token, no quota. The gated repos
(FLUX.1-schnell, FLUX.1-dev, stable-diffusion-3.5-*) are deliberately absent:
they return GatedRepoError 401 without a token and an accepted license, and this
project has nowhere to put a token. The flux tier uses ungated Apache-2.0
derivatives instead.

None of these pipelines load a safety checker: SDXL and flux pipelines don't
define one at all, and the SD1.5 pipeline is constructed with
`safety_checker=None`, so no output is filtered, blurred, or blacked out. There
is also no prompt-side keyword filtering anywhere in this codebase.

Memory columns are GiB (a 16 GB machine reports 16.0); download_gb is decimal GB.
`min_budget_gb` is GPU-visible memory and `min_ram_gb` is system RAM — for flux
they diverge sharply, because flux pipelines are loaded with
enable_model_cpu_offload(): one component is resident on the GPU at a time, while
system RAM still has to hold the whole model. Both must be satisfied.

min_budget_gb starts from the resident fp16 pipeline rather than from a round
number: every component is loaded at once (only flux offloads), so the floor is
the sum of the parameter counts times two bytes. On top of that goes an
allowance for attention-sliced activations, which grows with the entry's default
resolution — around 0.6-1.2 GiB at 512px, and deliberately generous at 1024px,
where underestimating costs more than excluding a borderline machine: an
out-of-memory condition on MPS sometimes hangs the process instead of raising.
The resident floors are:

    SD 1.5    UNet 860M + VAE 84M + CLIP-L 123M   = 1.07B  -> 1.99 GiB resident
    SD 2.1    UNet 866M + VAE 84M + OpenCLIP 340M = 1.29B  -> 2.40 GiB resident
    SDXL      UNet 2.57B + VAE 84M + two encoders = 3.47B  -> 6.46 GiB resident

That is what puts a real 4 GiB card — which reports a 3.6 GiB budget once
hardware.py subtracts its 10% headroom — on the running side of the SD 1.5 line
and the blocked side of the SDXL one.
"""

from dataclasses import dataclass, field, asdict


@dataclass
class ModelSpec:
    key: str
    name: str
    repo: str
    # The pipeline family, not the checkpoint lineage: "sd15" means
    # StableDiffusionPipeline, which is also how an SD 2.1 checkpoint loads.
    arch: str  # "sd15" | "sdxl" | "flux"
    download_gb: float
    blurb: str
    steps: int
    guidance: float
    width: int
    height: int
    min_budget_gb: float      # GPU-visible memory required, GiB
    min_ram_gb: float         # system RAM required, GiB
    baseline_seconds: float   # per image on the reference machine (M1 Pro, 16-core GPU)
    baseline_measured: bool   # False when extrapolated rather than timed
    quality_rank: int         # curated preference order, higher is better
    # sdxl-turbo style distilled models ignore CFG entirely
    supports_negative: bool = True
    # Distilled (turbo, LCM, flux) and EDM-trained (Playground) models ship a
    # scheduler that must survive; everything else converges better on DPM++.
    keep_scheduler: bool = False
    tags: list = field(default_factory=list)

    def to_json(self):
        return asdict(self)


MODELS = [
    # ---------------------------------------------------------------- sd15 ---
    ModelSpec(
        key="lcm-dreamshaper-7",
        name="LCM DreamShaper v7",
        repo="SimianLuo/LCM_Dreamshaper_v7",
        arch="sd15",
        download_gb=4.3,
        blurb="SD 1.5 distilled with Latent Consistency: usable images in 4-8 steps. "
        "The fastest way to a 768px image on a small machine. MIT.",
        steps=6,
        guidance=1.5,
        width=768,
        height=768,
        # 1.99 GiB resident, but its 768px default costs roughly 2.25x the
        # activations of a 512px render — the widest attention maps of the SD 1.5
        # entries, and the reason this sits above dreamshaper-8.
        min_budget_gb=3.5,
        min_ram_gb=8,
        baseline_seconds=4,
        baseline_measured=False,
        quality_rank=25,
        keep_scheduler=True,
        tags=["uncensored", "fastest", "small"],
    ),
    ModelSpec(
        key="dreamshaper-8",
        name="DreamShaper 8 (SD 1.5)",
        repo="Lykon/dreamshaper-8",
        arch="sd15",
        download_gb=2.1,
        blurb="Smallest download. SD1.5 lineage, very broad concept coverage, "
        "huge LoRA ecosystem. safety_checker is explicitly disabled.",
        steps=25,
        guidance=7.0,
        width=512,
        height=768,
        # The floor of the catalog: 1.99 GiB resident and under 3 GiB at 512x768.
        min_budget_gb=3.0,
        min_ram_gb=8,
        baseline_seconds=18,
        baseline_measured=True,
        quality_rank=40,
        tags=["uncensored", "fast", "small"],
    ),
    ModelSpec(
        key="sd-turbo",
        name="SD Turbo",
        repo="stabilityai/sd-turbo",
        # An SD 2.1 checkpoint, loaded by StableDiffusionPipeline like the SD 1.5
        # entries — the model_index declares a null safety_checker, which the
        # sd15 load path already passes explicitly.
        arch="sd15",
        download_gb=2.6,
        blurb="Stability's 1-4 step distillation of SD 2.1. The smallest memory "
        "footprint of the set — a 512px image in a couple of seconds on a 4 GB "
        "card. No safety checker, but the base was trained on filtered data, so "
        "it is more restricted in practice than the DreamShaper fine-tunes.",
        steps=2,
        guidance=0.0,
        width=512,
        height=512,
        # 2.40 GiB resident; 512px activations keep the peak near 3 GiB.
        min_budget_gb=3.0,
        min_ram_gb=8,
        baseline_seconds=2,
        baseline_measured=False,
        quality_rank=28,
        supports_negative=False,
        keep_scheduler=True,
        tags=["fastest", "small", "preview"],
    ),
    # ---------------------------------------------------------------- sdxl ---
    ModelSpec(
        key="sdxl-turbo",
        name="SDXL Turbo",
        repo="stabilityai/sdxl-turbo",
        arch="sdxl",
        download_gb=6.9,
        blurb="Stability's 1-4 step distilled model. Near-instant previews at 512px. "
        "Ignores negative prompts and CFG by design. Base model, lightly curated data.",
        steps=3,
        guidance=0.0,
        width=512,
        height=512,
        min_budget_gb=9.5,
        min_ram_gb=16,
        baseline_seconds=5,
        baseline_measured=False,
        quality_rank=30,
        supports_negative=False,
        keep_scheduler=True,
        tags=["fastest", "preview"],
    ),
    ModelSpec(
        key="dreamshaper-xl-turbo",
        name="DreamShaper XL v2 Turbo",
        repo="Lykon/dreamshaper-xl-v2-turbo",
        arch="sdxl",
        download_gb=6.9,
        blurb="SDXL quality at 1024px in 6-8 steps — the best quality-per-second "
        "of the set. Community fine-tune, no content filtering in weights or pipeline.",
        steps=7,
        guidance=2.0,
        width=1024,
        height=1024,
        min_budget_gb=9.5,
        min_ram_gb=16,
        baseline_seconds=40,
        baseline_measured=True,
        quality_rank=75,
        keep_scheduler=True,
        tags=["uncensored", "fast"],
    ),
    ModelSpec(
        key="juggernaut-xl-v9",
        name="Juggernaut XL v9",
        repo="RunDiffusion/Juggernaut-XL-v9",
        arch="sdxl",
        download_gb=7.1,
        blurb="Photorealism specialist. Slower (full 30-step CFG sampling) but the "
        "best skin/lighting/detail of the SDXL fine-tunes. Trained without filtering.",
        steps=30,
        guidance=6.0,
        width=1024,
        height=1024,
        min_budget_gb=9.5,
        min_ram_gb=16,
        baseline_seconds=180,
        baseline_measured=False,
        quality_rank=68,
        tags=["uncensored", "photoreal", "slow"],
    ),
    ModelSpec(
        key="realvis-xl-v4",
        name="RealVisXL V4.0",
        repo="SG161222/RealVisXL_V4.0",
        arch="sdxl",
        download_gb=6.9,
        blurb="The other well-known photoreal SDXL fine-tune. Stronger on people "
        "and portraits than Juggernaut; also unfiltered.",
        steps=30,
        guidance=6.0,
        width=1024,
        height=1024,
        min_budget_gb=9.5,
        min_ram_gb=16,
        baseline_seconds=180,
        baseline_measured=False,
        quality_rank=67,
        tags=["uncensored", "photoreal", "slow"],
    ),
    ModelSpec(
        key="playground-v25",
        name="Playground v2.5",
        repo="playgroundai/playground-v2.5-1024px-aesthetic",
        arch="sdxl",
        download_gb=7.0,
        blurb="SDXL architecture, trained from scratch for aesthetics — stronger "
        "color and composition than the fine-tunes. Ships an EDM scheduler it needs.",
        steps=30,
        guidance=3.0,
        width=1024,
        height=1024,
        min_budget_gb=9.5,
        min_ram_gb=16,
        baseline_seconds=200,
        baseline_measured=False,
        quality_rank=70,
        keep_scheduler=True,
        tags=["photoreal", "slow"],
    ),
    # ---------------------------------------------------------------- flux ---
    ModelSpec(
        key="flex-1-alpha",
        name="Flex.1 alpha",
        repo="ostris/Flex.1-alpha",
        arch="flux",
        download_gb=26.3,
        blurb="8B flux derivative, Apache-2.0, with a trained guidance embedder — "
        "unlike schnell it actually responds to guidance. Workstation memory only.",
        steps=50,
        guidance=3.5,
        width=1024,
        height=1024,
        min_budget_gb=20.0,
        min_ram_gb=36,
        baseline_seconds=300,
        baseline_measured=False,
        quality_rank=85,
        supports_negative=False,
        keep_scheduler=True,
        tags=["uncensored", "slow", "large"],
    ),
    ModelSpec(
        key="shuttle-3-diffusion",
        name="Shuttle 3 Diffusion",
        repo="shuttleai/shuttle-3-diffusion",
        arch="flux",
        download_gb=33.7,
        blurb="12B FLUX.1-schnell derivative, Apache-2.0. The best prompt adherence "
        "and text rendering of the set, in 4 steps. Workstation memory only.",
        steps=4,
        guidance=3.5,
        width=1024,
        height=1024,
        min_budget_gb=26.0,
        min_ram_gb=48,
        baseline_seconds=120,
        baseline_measured=False,
        quality_rank=90,
        supports_negative=False,
        keep_scheduler=True,
        tags=["uncensored", "large"],
    ),
]

BY_KEY = {m.key: m for m in MODELS}
# Used only when no hardware profile exists; otherwise the recommendation is
# computed from quality_rank among the models that fit.
DEFAULT_MODEL = "dreamshaper-xl-turbo"
