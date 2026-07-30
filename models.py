"""Model registry for local image generation on Apple Silicon.

Every entry runs fully offline after the first download. None of these pipelines
load a safety checker: SDXL pipelines don't define one at all, and the SD1.5
pipeline is constructed with `safety_checker=None`, so no output is filtered,
blurred, or blacked out. There is also no prompt-side keyword filtering anywhere
in this codebase.
"""

from dataclasses import dataclass, field, asdict


@dataclass
class ModelSpec:
    key: str
    name: str
    repo: str
    arch: str  # "sd15" | "sdxl"
    download_gb: float
    blurb: str
    steps: int
    guidance: float
    width: int
    height: int
    # sdxl-turbo style distilled models ignore CFG entirely
    supports_negative: bool = True
    speed: str = ""
    tags: list = field(default_factory=list)

    def to_json(self):
        return asdict(self)


MODELS = [
    ModelSpec(
        key="dreamshaper-xl-turbo",
        name="DreamShaper XL v2 Turbo",
        repo="Lykon/dreamshaper-xl-v2-turbo",
        arch="sdxl",
        download_gb=6.9,
        blurb="Best overall default for a 16 GB M1 Pro. SDXL quality at 1024px in "
        "6-8 steps. Community fine-tune, no content filtering in weights or pipeline.",
        steps=7,
        guidance=2.0,
        width=1024,
        height=1024,
        speed="~40 s / image (measured)",
        tags=["recommended", "uncensored", "fast"],
    ),
    ModelSpec(
        key="dreamshaper-8",
        name="DreamShaper 8 (SD 1.5)",
        repo="Lykon/dreamshaper-8",
        arch="sd15",
        download_gb=2.1,
        blurb="Smallest and fastest. SD1.5 lineage, very broad concept coverage, "
        "huge LoRA ecosystem. safety_checker is explicitly disabled.",
        steps=25,
        guidance=7.0,
        width=512,
        height=768,
        speed="~18 s / image (measured)",
        tags=["uncensored", "fastest", "small"],
    ),
    ModelSpec(
        key="juggernaut-xl-v9",
        name="Juggernaut XL v9",
        repo="RunDiffusion/Juggernaut-XL-v9",
        arch="sdxl",
        download_gb=7.1,
        blurb="Photorealism specialist. Slower (full 30-step CFG sampling) but the "
        "best skin/lighting/detail of the set. Trained without content filtering.",
        steps=30,
        guidance=6.0,
        width=1024,
        height=1024,
        speed="~3 min / image (estimated)",
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
        speed="~3 min / image (estimated)",
        tags=["uncensored", "photoreal", "slow"],
    ),
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
        supports_negative=False,
        speed="~5 s / image (estimated)",
        tags=["fastest", "preview"],
    ),
]

BY_KEY = {m.key: m for m in MODELS}
DEFAULT_MODEL = "dreamshaper-xl-turbo"
