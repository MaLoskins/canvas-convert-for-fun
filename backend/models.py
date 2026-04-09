"""
Model adapter layer for Live Canvas Art.

Each adapter wraps a specific Diffusers pipeline family (T2I-Adapter, ControlNet,
ControlNet Union) behind a common interface so the server can treat them uniformly.
"""

from __future__ import annotations

import gc
import io
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    ControlNetUnionModel,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLAdapterPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetUnionPipeline,
    T2IAdapter,
)
from PIL import Image, ImageOps, ImageStat

logger = logging.getLogger(__name__)

SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_VAE = "madebyollin/sdxl-vae-fp16-fix"
DEVICE = torch.device("cuda")
DTYPE = torch.float16


# ═══════════════════════════════════════════════════════════════════════
# Common image utilities
# ═══════════════════════════════════════════════════════════════════════

def resize_and_pad(img: Image.Image, target: int, fill: int = 0) -> Image.Image:
    """Resize preserving aspect ratio, centre-pad to target x target square."""
    w, h = img.size
    scale = target / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    padded = Image.new(img.mode, (target, target), fill)
    padded.paste(img, ((target - nw) // 2, (target - nh) // 2))
    return padded


def is_blank(img: Image.Image, threshold: float = 0.005) -> bool:
    """True if the image is essentially empty (mean brightness < threshold)."""
    stat = ImageStat.Stat(img.convert("L"))
    return (stat.mean[0] / 255.0) < threshold


def load_shared_vae() -> AutoencoderKL:
    logger.info("Loading shared SDXL VAE (fp16-fix)...")
    return AutoencoderKL.from_pretrained(SDXL_VAE, torch_dtype=DTYPE)


# ═══════════════════════════════════════════════════════════════════════
# Preprocessing strategies
# ═══════════════════════════════════════════════════════════════════════

def preprocess_invert_to_white_on_black(raw: bytes, size: int) -> Image.Image:
    """Black-on-white canvas -> white-on-black RGB.  Used by T2I-Adapter sketch/lineart."""
    img = Image.open(io.BytesIO(raw)).convert("L")
    img = ImageOps.invert(img)
    img = resize_and_pad(img, size, fill=0)
    return img.convert("RGB")


def preprocess_direct_scribble(raw: bytes, size: int) -> Image.Image:
    """Black-on-white canvas -> black-on-white RGB.  Faithful scribble pass-through."""
    img = Image.open(io.BytesIO(raw)).convert("L")
    img = resize_and_pad(img, size, fill=255)
    return img.convert("RGB")


def preprocess_canny_like(raw: bytes, size: int) -> Image.Image:
    """Black-on-white canvas -> thin white edges on black RGB.
    Invert, then threshold to clean binary for canny-style control."""
    img = Image.open(io.BytesIO(raw)).convert("L")
    img = ImageOps.invert(img)
    img = resize_and_pad(img, size, fill=0)
    # Threshold to clean binary edges
    img = img.point(lambda p: 255 if p > 30 else 0)
    return img.convert("RGB")


def preprocess_mistoline(raw: bytes, size: int) -> Image.Image:
    """MistoLine: robust across line types.  White lines on black, same as sketch."""
    return preprocess_invert_to_white_on_black(raw, size)


# ═══════════════════════════════════════════════════════════════════════
# Generation request (common across all adapters)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GenRequest:
    prompt: str
    negative_prompt: str
    image_bytes: bytes
    steps: int = 12
    guidance: float = 7.5
    conditioning_scale: float = 0.9
    seed: int | None = None
    size: int = 768
    # Union-specific
    union_mode: str | None = None


# ═══════════════════════════════════════════════════════════════════════
# Model registry entry
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ModelInfo:
    key: str
    name: str
    family: str  # t2i_adapter | controlnet | controlnet_union
    repo: str
    default_conditioning_scale: float = 0.9
    description: str = ""
    license_note: str = ""
    # For union: available sub-modes
    union_modes: dict[str, int] = field(default_factory=dict)
    # Display stats
    quality: str = ""       # e.g. "Very high", "High", "Medium-high"
    speed: str = ""         # e.g. "Fast", "Medium", "Medium-slow"
    adherence: str = ""     # e.g. "Very high", "High", "Medium"
    size: str = ""          # e.g. "~2.5 GB", "~474 MB"
    input_type: str = ""    # e.g. "Sketch", "Scribble / anyline"


MODEL_REGISTRY: dict[str, ModelInfo] = {
    "t2i_sketch": ModelInfo(
        key="t2i_sketch",
        name="T2I-Adapter Sketch",
        family="t2i_adapter",
        repo="TencentARC/t2i-adapter-sketch-sdxl-1.0",
        default_conditioning_scale=0.9,
        description="Sketch-conditioned T2I-Adapter for SDXL. White outlines on black.",
        quality="Medium-high", speed="Fast", adherence="Medium",
        size="~474 MB", input_type="Sketch",
    ),
    "t2i_lineart": ModelInfo(
        key="t2i_lineart",
        name="T2I-Adapter Lineart",
        family="t2i_adapter",
        repo="TencentARC/t2i-adapter-lineart-sdxl-1.0",
        default_conditioning_scale=0.8,
        description="Clean line-art conditioning when speed matters.",
        quality="Medium-high", speed="Fast", adherence="Medium-high",
        size="~474 MB", input_type="Line art",
    ),
    "mistoline": ModelInfo(
        key="mistoline",
        name="MistoLine",
        family="controlnet",
        repo="TheMistoAI/MistoLine",
        default_conditioning_scale=0.7,
        description="Best overall SDXL line-to-image fidelity. Adapts to any line art input type.",
        license_note="TheMistoAI/MistoLine - check repo for license and attribution requirements.",
        quality="Very high", speed="Medium-slow", adherence="Very high",
        size="~2.5 GB", input_type="Any line art",
    ),
    "cn_scribble": ModelInfo(
        key="cn_scribble",
        name="ControlNet Scribble",
        family="controlnet",
        repo="xinsir/controlnet-scribble-sdxl-1.0",
        default_conditioning_scale=0.7,
        description="Hand-drawn scribbles with strong adherence. Tolerates varied line widths.",
        quality="High", speed="Medium", adherence="High",
        size="~2.5 GB", input_type="Scribble / anyline",
    ),
    "cn_canny": ModelInfo(
        key="cn_canny",
        name="ControlNet Canny",
        family="controlnet",
        repo="xinsir/controlnet-canny-sdxl-1.0",
        default_conditioning_scale=0.7,
        description="Clean edge-map to image at SDXL scale. Best with thin clean edges.",
        quality="High", speed="Medium-slow", adherence="High",
        size="~5.0 GB", input_type="Canny edges",
    ),
    "cn_union": ModelInfo(
        key="cn_union",
        name="ControlNet Union",
        family="controlnet_union",
        repo="xinsir/controlnet-union-sdxl-1.0",
        default_conditioning_scale=0.7,
        description="One checkpoint for many control conditions. Broad utility.",
        quality="High", speed="Medium-slow", adherence="Variable",
        size="~5.1 GB", input_type="Multi-control",
        union_modes={
            "scribble": 2,
            "lineart": 3,
        },
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# Abstract adapter interface
# ═══════════════════════════════════════════════════════════════════════

class BaseAdapter(ABC):
    """Common interface for all model adapters."""

    def __init__(self, info: ModelInfo):
        self.info = info
        self.pipe = None

    @abstractmethod
    def load(self, vae: AutoencoderKL) -> None:
        """Load model weights and build the pipeline on GPU."""

    def unload(self) -> None:
        """Release GPU memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Adapter [%s] unloaded.", self.info.key)

    @property
    def is_loaded(self) -> bool:
        return self.pipe is not None

    @abstractmethod
    def preprocess(self, raw: bytes, size: int) -> Image.Image:
        """Convert raw canvas PNG bytes to the control image this model expects."""

    @abstractmethod
    def generate(self, req: GenRequest) -> Image.Image | None:
        """Run inference. Returns a PIL Image or None if the sketch is blank."""

    def _make_generator(self, seed: int | None) -> torch.Generator | None:
        if seed is not None and seed >= 0:
            return torch.Generator(device="cuda").manual_seed(seed)
        return None

    def _apply_scheduler_and_opts(self, pipe):
        """Set scheduler and optional optimisations."""
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("  xformers enabled for [%s].", self.info.key)
        except Exception:
            logger.info("  xformers not available for [%s], using default attention.", self.info.key)


# ═══════════════════════════════════════════════════════════════════════
# T2I-Adapter adapters
# ═══════════════════════════════════════════════════════════════════════

class T2IAdapterSketchAdapter(BaseAdapter):
    def load(self, vae):
        logger.info("Loading T2I-Adapter [%s] from %s ...", self.info.key, self.info.repo)
        adapter = T2IAdapter.from_pretrained(self.info.repo, torch_dtype=DTYPE, variant="fp16")
        pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            SDXL_BASE, adapter=adapter, vae=vae, torch_dtype=DTYPE, variant="fp16",
        )
        self._apply_scheduler_and_opts(pipe)
        pipe.to(DEVICE)
        self.pipe = pipe
        logger.info("T2I-Adapter [%s] ready.", self.info.key)

    def preprocess(self, raw, size):
        return preprocess_invert_to_white_on_black(raw, size)

    def generate(self, req):
        ctrl = self.preprocess(req.image_bytes, req.size)
        if is_blank(ctrl):
            return None
        prompt = req.prompt.strip() or "a drawing"
        with torch.inference_mode():
            return self.pipe(
                prompt=prompt,
                negative_prompt=req.negative_prompt or "ugly, blurry, low quality, distorted",
                image=ctrl,
                num_inference_steps=req.steps,
                adapter_conditioning_scale=req.conditioning_scale,
                guidance_scale=req.guidance,
                generator=self._make_generator(req.seed),
            ).images[0]


class T2IAdapterLineartAdapter(T2IAdapterSketchAdapter):
    """Same pipeline path as sketch, same preprocessing (white-on-black)."""
    pass


# ═══════════════════════════════════════════════════════════════════════
# Standard ControlNet adapters
# ═══════════════════════════════════════════════════════════════════════

class ControlNetAdapter(BaseAdapter):
    """Base for standard SDXL ControlNet models."""

    def load(self, vae):
        logger.info("Loading ControlNet [%s] from %s ...", self.info.key, self.info.repo)
        controlnet = ControlNetModel.from_pretrained(self.info.repo, torch_dtype=DTYPE)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            SDXL_BASE, controlnet=controlnet, vae=vae, torch_dtype=DTYPE, variant="fp16",
        )
        self._apply_scheduler_and_opts(pipe)
        pipe.to(DEVICE)
        self.pipe = pipe
        logger.info("ControlNet [%s] ready.", self.info.key)

    @abstractmethod
    def preprocess(self, raw, size):
        ...

    def generate(self, req):
        ctrl = self.preprocess(req.image_bytes, req.size)
        if is_blank(ctrl):
            return None
        prompt = req.prompt.strip() or "a drawing"
        with torch.inference_mode():
            return self.pipe(
                prompt=prompt,
                negative_prompt=req.negative_prompt or "ugly, blurry, low quality, distorted",
                image=ctrl,
                num_inference_steps=req.steps,
                controlnet_conditioning_scale=req.conditioning_scale,
                guidance_scale=req.guidance,
                generator=self._make_generator(req.seed),
            ).images[0]


class MistoLineAdapter(ControlNetAdapter):
    def load(self, vae):
        logger.info("Loading ControlNet [%s] from %s ...", self.info.key, self.info.repo)
        controlnet = ControlNetModel.from_pretrained(
            self.info.repo, torch_dtype=DTYPE, variant="fp16",
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            SDXL_BASE, controlnet=controlnet, vae=vae, torch_dtype=DTYPE, variant="fp16",
        )
        self._apply_scheduler_and_opts(pipe)
        pipe.to(DEVICE)
        self.pipe = pipe
        logger.info("ControlNet [%s] ready.", self.info.key)

    def preprocess(self, raw, size):
        return preprocess_mistoline(raw, size)


class ScribbleAdapter(ControlNetAdapter):
    def preprocess(self, raw, size):
        return preprocess_direct_scribble(raw, size)


class CannyAdapter(ControlNetAdapter):
    def preprocess(self, raw, size):
        return preprocess_canny_like(raw, size)


# ═══════════════════════════════════════════════════════════════════════
# ControlNet Union adapter
# ═══════════════════════════════════════════════════════════════════════

class ControlNetUnionAdapter(BaseAdapter):
    def load(self, vae):
        logger.info("Loading ControlNet Union [%s] from %s ...", self.info.key, self.info.repo)
        controlnet = ControlNetUnionModel.from_pretrained(
            self.info.repo,
            torch_dtype=DTYPE,
            config_file_name="config_promax.json",
        )
        pipe = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
            SDXL_BASE, controlnet=controlnet, vae=vae, torch_dtype=DTYPE, variant="fp16",
        )
        self._apply_scheduler_and_opts(pipe)
        pipe.to(DEVICE)
        self.pipe = pipe
        logger.info("ControlNet Union [%s] ready.", self.info.key)

    def _resolve_mode(self, req: GenRequest) -> int:
        mode_name = req.union_mode or "scribble"
        modes = self.info.union_modes
        if mode_name not in modes:
            logger.warning("Unknown union mode '%s', falling back to 'scribble'.", mode_name)
            mode_name = "scribble"
        return modes[mode_name]

    def preprocess(self, raw, size, mode_name: str = "scribble"):
        if mode_name == "lineart":
            return preprocess_canny_like(raw, size)
        return preprocess_direct_scribble(raw, size)

    def generate(self, req):
        mode_idx = self._resolve_mode(req)
        mode_name = req.union_mode or "scribble"
        ctrl = self.preprocess(req.image_bytes, req.size, mode_name)
        if is_blank(ctrl):
            return None
        prompt = req.prompt.strip() or "a drawing"
        with torch.inference_mode():
            return self.pipe(
                prompt=prompt,
                negative_prompt=req.negative_prompt or "ugly, blurry, low quality, distorted",
                control_image=ctrl,
                num_inference_steps=req.steps,
                controlnet_conditioning_scale=req.conditioning_scale,
                guidance_scale=req.guidance,
                generator=self._make_generator(req.seed),
                control_mode=mode_idx,
            ).images[0]


# ═══════════════════════════════════════════════════════════════════════
# Adapter factory
# ═══════════════════════════════════════════════════════════════════════

_ADAPTER_CLASSES: dict[str, type[BaseAdapter]] = {
    "t2i_sketch": T2IAdapterSketchAdapter,
    "t2i_lineart": T2IAdapterLineartAdapter,
    "mistoline": MistoLineAdapter,
    "cn_scribble": ScribbleAdapter,
    "cn_canny": CannyAdapter,
    "cn_union": ControlNetUnionAdapter,
}


def create_adapter(key: str) -> BaseAdapter:
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key: {key}")
    info = MODEL_REGISTRY[key]
    cls = _ADAPTER_CLASSES[key]
    return cls(info)


# ═══════════════════════════════════════════════════════════════════════
# Model manager — hot-swap with one model warm at a time
# ═══════════════════════════════════════════════════════════════════════

class ModelManager:
    """Manages loading / unloading of adapters.  Only one model is active on GPU."""

    def __init__(self):
        self._vae: AutoencoderKL | None = None
        self._active: BaseAdapter | None = None
        self._active_key: str | None = None
        self._loading: bool = False
        self._load_error: str | None = None

    @property
    def active_key(self) -> str | None:
        return self._active_key

    @property
    def is_ready(self) -> bool:
        return self._active is not None and self._active.is_loaded and not self._loading

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def active_adapter(self) -> BaseAdapter | None:
        return self._active

    def get_status(self) -> dict:
        if self._load_error:
            return {"status": "error", "message": self._load_error, "model": self._active_key}
        if self._loading:
            return {"status": "loading", "message": f"Loading {self._active_key}...", "model": self._active_key}
        if self.is_ready:
            return {"status": "ready", "model": self._active_key}
        return {"status": "idle", "message": "No model loaded.", "model": None}

    def load_model(self, key: str) -> None:
        """Synchronous. Call from a thread via run_in_executor."""
        if not torch.cuda.is_available():
            self._load_error = "CUDA is not available."
            return

        self._load_error = None
        self._loading = True

        try:
            # Unload previous
            if self._active is not None:
                logger.info("Unloading previous model [%s]...", self._active_key)
                self._active.unload()
                self._active = None

            # Shared VAE (kept on CPU, the pipeline moves it to GPU)
            if self._vae is None:
                self._vae = load_shared_vae()

            adapter = create_adapter(key)
            adapter.load(self._vae)
            self._active = adapter
            self._active_key = key
            logger.info("Model [%s] active.", key)

        except Exception as e:
            self._load_error = f"Failed to load model '{key}': {e}"
            logger.exception(self._load_error)
            self._active = None
            self._active_key = key  # keep key so UI knows which model failed

        finally:
            self._loading = False

    def generate(self, req: GenRequest) -> Image.Image | None:
        if not self.is_ready or self._active is None:
            return None
        return self._active.generate(req)
