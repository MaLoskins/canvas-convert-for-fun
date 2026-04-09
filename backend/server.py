"""
Live Canvas Art - Backend Server
T2I-Adapter Sketch SDXL pipeline with WebSocket API for live sketch-to-image generation.
"""

import asyncio
import base64
import io
import logging
import time
from contextlib import asynccontextmanager

import torch
from diffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
    AutoencoderKL,
)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global pipeline holder
# ---------------------------------------------------------------------------
pipeline = None
pipeline_ready = asyncio.Event()
load_error: str | None = None

PREVIEW_STEPS = 12          # fast preview
PREVIEW_SIZE = 768           # resolution for interactive previews
HQ_STEPS = 30               # high-quality render
HQ_SIZE = 1024               # high-quality resolution


def load_pipeline():
    """Load the full model stack once. Called at startup in a background thread."""
    global pipeline, load_error

    if not torch.cuda.is_available():
        load_error = "CUDA is not available. This application requires an NVIDIA GPU with CUDA support."
        logger.error(load_error)
        return

    device = torch.device("cuda")
    dtype = torch.float16
    logger.info("Loading T2I-Adapter sketch model...")

    try:
        adapter = T2IAdapter.from_pretrained(
            "TencentARC/t2i-adapter-sketch-sdxl-1.0",
            torch_dtype=dtype,
            variant="fp16",
        )

        logger.info("Loading SDXL VAE (fp16-fix)...")
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=dtype,
        )

        logger.info("Loading Stable Diffusion XL base pipeline...")
        pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            adapter=adapter,
            vae=vae,
            torch_dtype=dtype,
            variant="fp16",
        )

        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(device)

        # Enable memory-efficient attention if xformers is available
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("xformers memory-efficient attention enabled.")
        except Exception:
            logger.info("xformers not available, using default attention.")

        pipeline = pipe
        logger.info("Pipeline loaded and ready on %s", device)

    except Exception as e:
        load_error = f"Failed to load pipeline: {e}"
        logger.exception(load_error)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start model loading in a background thread so the server starts immediately
    loop = asyncio.get_event_loop()

    async def _load():
        await loop.run_in_executor(None, load_pipeline)
        pipeline_ready.set()

    asyncio.create_task(_load())
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health / status endpoint
# ---------------------------------------------------------------------------
@app.get("/status")
async def status():
    if load_error:
        return {"status": "error", "message": load_error}
    if pipeline is None:
        return {"status": "loading", "message": "Model is loading..."}
    return {"status": "ready"}


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def prepare_sketch_image(image_bytes: bytes, target_size: int) -> Image.Image:
    """
    Convert the frontend canvas to the adapter's expected format:
    white lines on a black background, single-channel, resized to target_size
    while preserving aspect ratio (padded to square).

    The frontend sends a PNG where the drawing is black strokes on a white
    (or transparent) background.  We need to invert this.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L")

    # Invert: black strokes on white -> white strokes on black
    from PIL import ImageOps
    img = ImageOps.invert(img)

    # Resize preserving aspect ratio, pad to square
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Pad to target_size x target_size centered
    padded = Image.new("L", (target_size, target_size), 0)
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    padded.paste(img, (offset_x, offset_y))

    # Convert to RGB (adapter expects 3-channel)
    return padded.convert("RGB")


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def is_blank_sketch(img: Image.Image, threshold: float = 0.005) -> bool:
    """Check if the sketch is essentially blank (all black after inversion)."""
    from PIL import ImageStat
    stat = ImageStat.Stat(img.convert("L"))
    # mean pixel value — if very low, sketch is blank
    return (stat.mean[0] / 255.0) < threshold


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(
    prompt: str,
    sketch_bytes: bytes,
    negative_prompt: str = "",
    steps: int = PREVIEW_STEPS,
    guidance: float = 7.5,
    adapter_scale: float = 0.9,
    seed: int | None = None,
    size: int = PREVIEW_SIZE,
) -> str | None:
    """Run a single inference pass. Returns base64-encoded JPEG or None."""
    if pipeline is None:
        return None

    sketch_img = prepare_sketch_image(sketch_bytes, size)

    if is_blank_sketch(sketch_img):
        return None

    if not prompt.strip():
        prompt = "a drawing"

    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    with torch.inference_mode():
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or "ugly, blurry, low quality, distorted",
            image=sketch_img,
            num_inference_steps=steps,
            adapter_conditioning_scale=adapter_scale,
            guidance_scale=guidance,
            generator=generator,
            num_images_per_prompt=1,
        ).images[0]

    return image_to_base64(result)


# ---------------------------------------------------------------------------
# WebSocket endpoint for live generation
#
# Protocol (request-reply, no queue):
#   1. Client connects, server sends {"type":"status", ...} periodically
#   2. When pipeline is ready, server sends {"type":"ready_for_next"}
#   3. Client sends {"type":"generate", prompt, image, ...} with latest state
#   4. Server sends {"type":"generating"}, runs inference, sends {"type":"result"}
#   5. Server sends {"type":"ready_for_next"} — go to step 3
#
# The frontend holds back until it receives ready_for_next, so at most one
# generation is in flight at any time.  No queue, no cancellation, no races.
# ---------------------------------------------------------------------------
@app.websocket("/ws/generate")
async def ws_generate(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket client connected")

    loop = asyncio.get_event_loop()
    generation_id = 0

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type", "generate")

            if msg_type == "ping":
                await ws.send_json({"type": "pong"})
                continue

            if msg_type == "status":
                if load_error:
                    await ws.send_json({"type": "status", "status": "error", "message": load_error})
                elif pipeline is None:
                    await ws.send_json({"type": "status", "status": "loading", "message": "Model is loading..."})
                else:
                    await ws.send_json({"type": "status", "status": "ready"})
                    # Also tell client it can send work
                    await ws.send_json({"type": "ready_for_next"})
                continue

            if msg_type != "generate":
                continue

            # --- Generate request ---
            prompt = data.get("prompt", "")
            image_b64 = data.get("image", "")
            negative_prompt = data.get("negative_prompt", "")
            steps = int(data.get("steps", PREVIEW_STEPS))
            guidance = float(data.get("guidance", 7.5))
            adapter_scale = float(data.get("adapter_scale", 0.9))
            seed = data.get("seed", None)
            if seed is not None:
                seed = int(seed)
            size = int(data.get("size", PREVIEW_SIZE))

            if not image_b64:
                await ws.send_json({"type": "ready_for_next"})
                continue

            if pipeline is None:
                await ws.send_json({
                    "type": "status",
                    "status": "loading",
                    "message": "Model still loading, please wait...",
                })
                continue

            generation_id += 1
            gid = generation_id
            image_bytes = base64.b64decode(image_b64)

            await ws.send_json({"type": "generating", "generation_id": gid})

            try:
                t0 = time.perf_counter()
                result_b64 = await loop.run_in_executor(
                    None, run_inference, prompt, image_bytes, negative_prompt,
                    steps, guidance, adapter_scale, seed, size,
                )
                elapsed = time.perf_counter() - t0

                if result_b64:
                    await ws.send_json({
                        "type": "result",
                        "image": result_b64,
                        "generation_id": gid,
                        "elapsed": round(elapsed, 2),
                    })
                else:
                    await ws.send_json({
                        "type": "skipped",
                        "generation_id": gid,
                        "reason": "blank_sketch",
                    })
            except Exception as e:
                logger.exception("Generation %d failed", gid)
                await ws.send_json({
                    "type": "error",
                    "message": str(e),
                    "generation_id": gid,
                })

            # Signal client to send next frame
            await ws.send_json({"type": "ready_for_next"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception("WebSocket error: %s", e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8188, log_level="info")
